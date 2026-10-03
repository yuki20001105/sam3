from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from annotation_manager import list_images, load_annotations, validate_annotations
from class_config import ClassConfigError, add_class, get_enabled_classes, get_class_by_name, load_classes
from config import DEFAULT_CHECKPOINT, DEFAULT_CLASSES_PATH, DEFAULT_DATASET_DIR, DEFAULT_IMAGE_DIR
from exporters import save_annotations
from image_utils import (
    canvas_prompt_objects,
    display_size,
    display_to_original_box,
    display_to_original_point,
    draw_annotations,
    merge_same_class_annotations,
    original_to_display_box,
)
from sam_predictor import Sam31Predictor
from state_manager import load_draft, save_draft

UNDO_LIMIT = 5
STATUS_LABELS = {"unprocessed": "未処理", "needs_review": "要確認", "approved": "確定済み"}

st.set_page_config(page_title="SAM 3.1 Annotation Tool", layout="wide")


@st.cache_data(show_spinner=False)
def cached_image_paths(image_dir: str) -> tuple[str, ...]:
    return tuple(str(path) for path in list_images(Path(image_dir)))


@st.cache_resource(show_spinner=False)
def get_predictor(checkpoint_path: str) -> Sam31Predictor:
    return Sam31Predictor(Path(checkpoint_path))


def count_approved(dataset_dir: str, revision: int) -> int:
    """Count approved drafts directly so status metrics never show stale data."""
    drafts_dir = Path(dataset_dir) / "drafts"
    if not drafts_dir.is_dir():
        return 0
    count = 0
    for path in drafts_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") == "approved":
            count += 1
    return count


def status_counts(dataset_dir: Path, images: list[Path]) -> dict[str, int]:
    counts = {"unprocessed": 0, "needs_review": 0, "approved": 0}
    for path in images:
        draft = load_draft(dataset_dir, path.name)
        status = (draft or {}).get("status", "unprocessed")
        counts[status if status in counts else "unprocessed"] += 1
    return counts


def next_pending_index(images: list[Path], dataset_dir: Path, current_index: int) -> int | None:
    """Return the next non-approved image, preferring images after the current one."""
    order = list(range(current_index + 1, len(images))) + list(range(0, current_index))
    for index in order:
        draft = load_draft(dataset_dir, images[index].name)
        if (draft or {}).get("status", "unprocessed") != "approved":
            return index
    return None


def processed_classes_from_draft(draft: dict | None) -> set[str]:
    """Read batch progress from both old (`prompt: null`) and current drafts."""
    prompt = (draft or {}).get("prompt")
    if not isinstance(prompt, dict):
        return set()
    classes = prompt.get("classes")
    if not isinstance(classes, (list, tuple, set)):
        return set()
    return {str(name) for name in classes}


def annotations_from_draft(draft: dict | None) -> list[dict]:
    annotations = (draft or {}).get("annotations")
    return annotations if isinstance(annotations, list) else []


def image_state_key(dataset_dir: Path, image_path: Path) -> str:
    """Keep Streamlit state isolated when the user switches datasets."""
    return f"{dataset_dir.resolve()}::{image_path.resolve()}"


def get_image_state(image_path: Path, dataset_dir: Path, classes: list[dict]) -> list[dict]:
    key = image_state_key(dataset_dir, image_path)
    if st.session_state.get("loaded_image") != key:
        with Image.open(image_path) as source:
            st.session_state["image_size"] = source.size
        loaded_annotations = load_annotations(
            dataset_dir, image_path, *st.session_state["image_size"], classes
        )
        for annotation in loaded_annotations:
            annotation.setdefault("annotation_id", uuid.uuid4().hex)
        st.session_state["annotations"] = loaded_annotations
        st.session_state["loaded_image"] = key
        st.session_state["selected_annotation_id"] = None
        st.session_state["prompt_points"] = []
        st.session_state["prompt_point_labels"] = []
        st.session_state["prompt_box"] = None
        st.session_state["prompt_owner"] = None
        st.session_state["undo_stack"] = []
    return st.session_state.setdefault("annotations", [])


def ensure_prompt_scope(mode: str, target_class: str | None, image_key: str) -> None:
    """Reset the geometric prompt buffer whenever image / mode / target class changes,
    so a Prompt drawn for one image or class is never reused on another."""
    owner = (image_key, mode, target_class)
    if st.session_state.get("prompt_owner") != owner:
        st.session_state["prompt_points"] = []
        st.session_state["prompt_point_labels"] = []
        st.session_state["prompt_box"] = None
        bump_canvas_generation()
        st.session_state["prompt_owner"] = owner


def bump_canvas_generation() -> None:
    for key in list(st.session_state):
        if str(key).startswith("prompt_canvas_initial::"):
            del st.session_state[key]
    st.session_state["canvas_generation"] = st.session_state.get("canvas_generation", 0) + 1


def push_undo(image_key: str, annotations: list[dict]) -> None:
    stack = st.session_state.setdefault("undo_stack", [])
    stack.append((image_key, copy.deepcopy(annotations)))
    del stack[:-UNDO_LIMIT]


def pop_undo(image_key: str) -> list[dict] | None:
    stack = st.session_state.setdefault("undo_stack", [])
    while stack:
        snapshot_key, snapshot = stack.pop()
        if snapshot_key == image_key:
            return snapshot
    return None


def save_current_draft(dataset_dir: Path, image_path: Path, image: Image.Image, annotations: list[dict]) -> bool:
    try:
        save_draft(dataset_dir, image_path.name, image.width, image.height, annotations, status="needs_review")
        return True
    except OSError as error:
        st.error(f"下書きの保存に失敗しました: {error}")
        return False


@st.fragment
def render_sam_add_tool(
    image_path: Path,
    image: Image.Image,
    dataset_dir: Path,
    checkpoint: Path,
    classes: list[dict],
    selected_names: list[str],
    available_names: list[str],
    confidence: float,
    image_key: str,
    annotations: list[dict],
) -> None:
    """Render Point/Box input in an isolated fragment.

    Canvas updates rerun only this fragment.  In particular, consecutive Point
    clicks no longer redraw the sidebar, result list, and the rest of the page.
    """
    st.subheader("不足しているBBoxをSAMで追加")
    target_col, mode_col = st.columns([1, 1.4])
    target_options = selected_names or available_names
    if st.session_state.get("sam_add_target_class") not in target_options:
        st.session_state["sam_add_target_class"] = target_options[0] if target_options else None
    with target_col:
        target_name = st.selectbox(
            "追加するクラス",
            target_options,
            key="sam_add_target_class",
            format_func=lambda name: get_class_by_name(classes, name)["display_name"],
        ) if target_options else None
    with mode_col:
        advanced_mode = st.radio(
            "指定方法",
            ["Point", "Box"],
            key="advanced_submode",
            horizontal=True,
            format_func=lambda mode: "ポイントをクリック" if mode == "Point" else "矩形で囲む",
        )
    ensure_prompt_scope(advanced_mode, target_name, image_key)

    point_type = "Positive"
    if advanced_mode == "Point":
        st.info("対象の内側を必要なだけ続けてクリックします。緑は対象、赤は除外する場所です。")
        point_kind = st.radio(
            "クリックの種類",
            ["Positive", "Negative"],
            key="point_input_kind",
            horizontal=True,
            format_func=lambda value: "対象点（緑）" if value == "Positive" else "除外点（赤）",
        )
        point_type = point_kind
    else:
        st.info("対象を左上から右下へドラッグして囲みます。最後に描いた枠を使用します。")

    original_size = image.size
    prompt_display_size = display_size(image)
    prompt_image = image.resize(prompt_display_size)
    point_color = "#22c55e" if advanced_mode != "Point" or point_type == "Positive" else "#ef4444"
    canvas_generation = st.session_state.get("canvas_generation", 0)
    initial_state_key = f"prompt_canvas_initial::{image_key}::{advanced_mode}::{canvas_generation}"
    if initial_state_key not in st.session_state:
        # Keep initialDrawing stable while the user adds consecutive points.
        # Rebuilding it from every component result makes the drawable-canvas
        # reload itself and can cause a rerun feedback loop.
        st.session_state[initial_state_key] = canvas_prompt_objects(
            st.session_state.get("prompt_points", []),
            st.session_state.get("prompt_point_labels", []),
            st.session_state.get("prompt_box"),
            original_size,
            prompt_display_size,
        )
    canvas_col, action_col = st.columns([3.2, 1])
    with canvas_col:
        canvas = st_canvas(
            background_image=prompt_image,
            initial_drawing=st.session_state[initial_state_key],
            drawing_mode="point" if advanced_mode == "Point" else "rect",
            fill_color="rgba(0,0,0,0)" if advanced_mode == "Box" else point_color,
            stroke_color="#facc15" if advanced_mode == "Box" else point_color,
            stroke_width=3,
            point_display_radius=7,
            update_streamlit=True,
            display_toolbar=False,
            width=prompt_display_size[0],
            height=prompt_display_size[1],
            key=f"prompt_canvas_{image_path.stem}_{advanced_mode}_{canvas_generation}",
        )
        update_prompt_from_canvas(canvas, advanced_mode, original_size, prompt_display_size)

    with action_col:
        st.markdown("#### 入力")
        if advanced_mode == "Point":
            labels = st.session_state.get("prompt_point_labels", [])
            positive_count = labels.count(1)
            negative_count = labels.count(0)
            total_count = positive_count + negative_count
            has_prompt = positive_count > 0
            st.markdown(f"**対象 {positive_count}点**  ・  除外 {negative_count}点")
            if has_prompt:
                st.success("続けてクリックできます")
            else:
                st.warning("対象をクリックしてください")
            button_label = f"この{total_count}点でBBoxを追加" if total_count else "ポイントからBBoxを追加"
        else:
            has_prompt = st.session_state.get("prompt_box") is not None
            if has_prompt:
                st.success("囲みを受け付けました")
            else:
                st.warning("対象をドラッグで囲んでください")
            button_label = "この囲みからBBoxを追加"

        run_button = st.button(
            button_label,
            type="primary",
            use_container_width=True,
            disabled=not has_prompt or not target_name,
            help="指定した位置からSAM 3.1でBBox候補を作成します。",
        )
        if advanced_mode == "Point":
            undo_point = st.button(
                "最後の1点を取り消す",
                use_container_width=True,
                disabled=not st.session_state.get("prompt_points"),
            )
            clear_points = st.button(
                "すべてクリア",
                use_container_width=True,
                disabled=not st.session_state.get("prompt_points"),
            )
            if undo_point:
                st.session_state["prompt_points"] = st.session_state.get("prompt_points", [])[:-1]
                st.session_state["prompt_point_labels"] = st.session_state.get("prompt_point_labels", [])[:-1]
                bump_canvas_generation()
                st.rerun(scope="fragment")
            if clear_points:
                st.session_state["prompt_points"] = []
                st.session_state["prompt_point_labels"] = []
                bump_canvas_generation()
                st.rerun(scope="fragment")
        else:
            clear_box = st.button("囲みをクリア", use_container_width=True, disabled=not has_prompt)
            if clear_box:
                st.session_state["prompt_box"] = None
                bump_canvas_generation()
                st.rerun(scope="fragment")

    if not run_button:
        return
    if not checkpoint.is_file():
        st.error(f"Checkpoint not found: {checkpoint}")
        return
    if advanced_mode == "Point" and not st.session_state.get("prompt_points"):
        st.error("画像内の対象を1回以上クリックしてください。")
        return
    if advanced_mode == "Box" and not st.session_state.get("prompt_box"):
        st.error("画像内の対象をドラッグして囲んでください。")
        return
    if not target_name:
        st.error("追加する対象クラスを選択してください。")
        return

    push_undo(image_key, annotations)
    progress = st.progress(0, text="Preparing SAM 3.1 model...")
    status = st.empty()
    predictor = get_predictor(str(checkpoint))
    target_config = get_class_by_name(classes, target_name)
    progress.progress(0.4, text=f"{target_config['display_name']}: running")
    if advanced_mode == "Point":
        predictions = predictor.predict_points(
            image,
            st.session_state["prompt_points"],
            st.session_state["prompt_point_labels"],
            confidence,
        )
        source = "sam3.1_point"
    else:
        box = st.session_state["prompt_box"]
        if box[2] <= box[0] or box[3] <= box[1]:
            st.error("Prompt box width and height must be positive.")
            return
        predictions = predictor.predict_box(image, box, confidence)
        source = "sam3.1_box"
    for annotation in predictions:
        annotation["class_id"] = target_config["id"]
        annotation["class_name"] = target_config["name"]
        annotation["source"] = source
        annotation["human_verified"] = False
    merged = merge_same_class_annotations(annotations, predictions)
    progress.progress(1.0, text="Integrating BBoxes")
    status.success(f"SAM 3.1 complete: {len(predictions)} BBoxes")
    st.session_state["annotations"] = merged
    if save_current_draft(dataset_dir, image_path, image, merged):
        if predictions:
            st.session_state["prompt_points"] = []
            st.session_state["prompt_point_labels"] = []
            st.session_state["prompt_box"] = None
            st.session_state["pending_active_tool"] = "非表示"
            st.session_state["flash_message"] = (
                f"{target_config['display_name']} のBBox候補を{len(predictions)}件追加しました。"
            )
            bump_canvas_generation()
            st.rerun()
        else:
            st.warning("この指定ではBBoxが見つかりませんでした。点を追加するか、Box指定をお試しください。")


def update_prompt_from_canvas(canvas_data, mode: str, original: tuple[int, int], display: tuple[int, int]) -> None:
    if not canvas_data or not canvas_data.json_data:
        return
    objects = canvas_data.json_data.get("objects", [])
    if mode == "Point":
        points, labels = [], []
        for item in objects:
            if item.get("type") != "circle":
                continue
            radius = float(item.get("radius", 0)) * float(item.get("scaleX", 1))
            x = float(item.get("left", 0)) + radius
            y = float(item.get("top", 0)) + radius
            points.append(display_to_original_point((x, y), original, display))
            labels.append(0 if "ef4444" in str(item.get("fill", "")).lower() else 1)
        st.session_state["prompt_points"] = points
        st.session_state["prompt_point_labels"] = labels
    elif mode == "Box":
        rectangles = [item for item in objects if item.get("type") == "rect"]
        if rectangles:
            item = rectangles[-1]
            x1 = float(item.get("left", 0))
            y1 = float(item.get("top", 0))
            x2 = x1 + float(item.get("width", 0)) * float(item.get("scaleX", 1))
            y2 = y1 + float(item.get("height", 0)) * float(item.get("scaleY", 1))
            st.session_state["prompt_box"] = display_to_original_box([x1, y1, x2, y2], original, display)


def predict_selected_classes(
    predictor: Sam31Predictor,
    image: Image.Image,
    selected_classes: list[dict],
    confidence: float,
    progress: st.delta_generator.DeltaGenerator,
    status: st.delta_generator.DeltaGenerator,
    completed_steps: int,
    total_steps: int,
) -> tuple[list[dict], int]:
    annotations: list[dict] = []
    for class_config in selected_classes:
        completed_steps += 1
        status.write(
            f"{class_config['display_name']}: running ({completed_steps} / {total_steps})"
        )
        candidates = predictor.predict_text(image, class_config["prompt"], confidence)
        for candidate in candidates:
            candidate["class_id"] = class_config["id"]
            candidate["class_name"] = class_config["name"]
            candidate["source"] = "sam3.1_text"
        annotations.extend(candidates)
        progress.progress(
            completed_steps / total_steps,
            text=f"{class_config['display_name']}: complete ({completed_steps} / {total_steps})",
        )
    return annotations, completed_steps


def run_prompt_for_image(
    predictor: Sam31Predictor,
    image: Image.Image,
    prompt_spec: dict,
    confidence: float,
) -> list[dict]:
    mode = prompt_spec["mode"]
    if mode == "Text":
        results: list[dict] = []
        for class_config in prompt_spec["classes"]:
            class_results = predictor.predict_text(image, class_config["prompt"], confidence)
            for result in class_results:
                result["class_id"] = class_config["id"]
                result["class_name"] = class_config["name"]
                result["source"] = "sam3.1_text"
            results.extend(class_results)
        return results

    # Point / Box are pure geometric prompts: the model is not told the class
    # name. The target class is only used to label the resulting BBox.
    class_config = prompt_spec["class"]
    if mode == "Point":
        results = predictor.predict_points(image, prompt_spec["points"], prompt_spec["labels"], confidence)
        source = "sam3.1_point"
    else:
        results = predictor.predict_box(image, prompt_spec["box"], confidence)
        source = "sam3.1_box"
    for result in results:
        result["class_id"] = class_config["id"]
        result["class_name"] = class_config["name"]
        result["source"] = source
        result["human_verified"] = False
    return results



def main() -> None:
    st.title("アノテーション確認")
    st.caption("候補はSAM 3.1が作成します。画像を確認し、問題がなければ確定するだけです。")
    flash_message = st.session_state.pop("flash_message", None)
    if flash_message:
        st.success(flash_message)

    with st.sidebar:
        st.header("自動アノテーション")
        st.caption("① 対象を設定　② 候補を一括作成　③ 画像ごとに確認")

        with st.expander("管理者向け：データ設定", expanded=False):
            image_dir = Path(st.text_input("Image directory", str(DEFAULT_IMAGE_DIR)))
            dataset_dir = Path(st.text_input("Output dataset", str(DEFAULT_DATASET_DIR)))
            checkpoint = Path(st.text_input("SAM 3.1 checkpoint", str(DEFAULT_CHECKPOINT)))
            classes_path = Path(st.text_input("Classes YAML", str(DEFAULT_CLASSES_PATH)))

        try:
            classes = load_classes(classes_path)
        except ClassConfigError as error:
            st.error(str(error))
            return

        images = [Path(path) for path in cached_image_paths(str(image_dir))]
        if not images:
            st.warning("No images found in the selected directory.")
            st.info("Supported: .jpg, .jpeg, .png, .bmp")
            return

        current_index = st.session_state.get("image_index", 0)
        current_index = max(0, min(int(current_index), len(images) - 1))
        image_path = images[current_index]

        available_names = [item["name"] for item in classes]
        if "selected_classes" not in st.session_state:
            st.session_state["selected_classes"] = [
                item["name"] for item in get_enabled_classes(classes)
            ]
        pending_class = st.session_state.pop("pending_added_class", None)
        if pending_class and pending_class in available_names:
            selected = [
                name for name in st.session_state["selected_classes"]
                if name in available_names
            ]
            if pending_class not in selected:
                selected.append(pending_class)
            st.session_state["selected_classes"] = selected

        with st.expander("① 対象クラスを確認・変更", expanded=False):
            select_all, clear_all = st.columns(2)
            if select_all.button("すべて選択"):
                st.session_state["selected_classes"] = available_names
                st.rerun()
            if clear_all.button("選択解除"):
                st.session_state["selected_classes"] = []
                st.rerun()
            selected_names = st.multiselect(
                "検出するクラス", options=available_names, key="selected_classes",
                format_func=lambda name: get_class_by_name(classes, name)["display_name"],
            )
        selected_labels = [get_class_by_name(classes, name)["display_name"] for name in selected_names]
        st.caption(f"対象クラス {len(selected_names)}件: " + "、".join(selected_labels[:4]) + ("…" if len(selected_labels) > 4 else ""))

        # Pre-fetch the current Point/Box mode & target class from session_state
        # (the actual widgets live next to the canvas, not in the sidebar, so the
        # Queue section below needs to read the values without creating them).
        advanced_mode = st.session_state.get("advanced_submode", "Point")
        target_name = st.session_state.get("sam_add_target_class")
        if target_name not in available_names and available_names:
            target_name = available_names[0]

        with st.expander("クラスを追加", expanded=False):
            new_class_name = st.text_input("New class name", key="new_class_name")
            new_class_enabled = st.checkbox("Enabled by default", value=True, key="new_class_enabled")
            if st.button("Add class to YAML"):
                try:
                    created = add_class(classes_path, new_class_name, new_class_enabled)
                    if created["enabled"]:
                        st.session_state["pending_added_class"] = created["name"]
                    st.success(f"Added {created['name']} (id={created['id']}). Existing IDs are never reassigned.")
                    st.rerun()
                except ClassConfigError as error:
                    st.error(str(error))

        counts = status_counts(dataset_dir, images)
        st.subheader("② 候補をまとめて作成")
        st.caption(f"未処理 {counts['unprocessed']}件 / 要確認 {counts['needs_review']}件 / 確定 {counts['approved']}件")
        batch_unprocessed_button = st.button(
            "未確定画像の候補を一括作成",
            type="primary",
            use_container_width=True,
            help="選択した対象クラスで、まだ確定していない画像すべてに候補BBoxを作成します（下書き保存のみ、自動確定しません）",
        )

        confidence = st.session_state.get("confidence", 0.5)
        overwrite_batch = False
        queue_prompt_button = False
        run_queue_button = False
        with st.expander("詳細設定", expanded=False):
            st.caption(
                "Textで見つからない対象は、画像上の「操作ツール」から Point / Box で直接指定できます。"
                "この指定はモデルにクラス名を伝えるものではなく、結果に付けるラベルとしてのみ使われます。"
            )
            confidence = st.slider(
                "Confidence", 0.0, 1.0, st.session_state.get("confidence", 0.5), 0.01, key="confidence",
                help="次回の推論で使うしきい値です。変更しただけでは既存のBBoxは消えません。",
            )
            st.caption(f"現在の指定方法: {advanced_mode} / 対象クラス: {target_name or '(未選択)'}")

            overwrite_batch = st.checkbox(
                "確定済み・下書きも上書き対象にする", value=False,
                help="オフの場合、確定済み(approved)の画像はスキップされます。",
            )
            st.divider()
            st.caption("Point / Box キュー（上級者向け）")
            queue_prompt_button = st.button("この画像の指定をキューに登録")
            run_queue_button = st.button("キューを一括実行")
            st.caption(f"登録済み: {len(st.session_state.get('prompt_queue', {}))}件")
            st.caption(f"実行環境: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    image_key = image_state_key(dataset_dir, image_path)
    annotations = get_image_state(image_path, dataset_dir, classes)
    image = Image.open(image_path).convert("RGB")
    draft = load_draft(dataset_dir, image_path.name)
    image_status = draft.get("status", "unprocessed") if draft else "unprocessed"
    ensure_prompt_scope(advanced_mode, target_name, image_key)

    # --- Current image and progress ---------------------------------------
    approved_count = count_approved(str(dataset_dir), st.session_state.get("approved_revision", 0))
    status_icon = {"unprocessed": "⚪", "needs_review": "🟠", "approved": "🟢"}.get(image_status, "⚪")
    header_left, header_status, header_count = st.columns([2.2, 1, 1])
    header_left.markdown(f"### {image_path.name}")
    header_left.caption(f"画像 {current_index + 1} / {len(images)}")
    header_status.markdown("**現在の状態**")
    header_status.markdown(f"{status_icon} **{STATUS_LABELS.get(image_status, image_status)}**")
    header_count.metric("確認の進捗", f"{approved_count} / {len(images)}")
    progress_ratio = min(1.0, max(0.0, approved_count / max(1, len(images))))
    remaining_count = max(0, len(images) - approved_count)
    st.progress(progress_ratio, text=f"確定済み {approved_count}件・残り {remaining_count}件")

    action_text = "候補を自動作成" if not annotations else "候補を再検索・追加"
    action_help = "選択中の対象クラスをSAM 3.1で検索し、候補BBoxを下書きへ追加します。"
    try_button = st.button(
        action_text,
        type="primary" if image_status == "unprocessed" and not annotations else "secondary",
        help=action_help,
    )
    if image_status == "unprocessed" and not annotations:
        st.info("この画像は未処理です。上のボタンで候補を作成するか、左側から全画像を一括処理してください。")

    if try_button:
        if not checkpoint.is_file():
            st.error(f"Checkpoint not found: {checkpoint}")
        elif not selected_names:
            st.error("Select at least one class.")
        else:
            push_undo(image_key, annotations)
            progress = st.progress(0, text="Preparing SAM 3.1 model...")
            status = st.empty()
            predictor = get_predictor(str(checkpoint))
            selected_classes = [get_class_by_name(classes, name) for name in selected_names]
            predictions, _ = predict_selected_classes(
                predictor, image, selected_classes, confidence, progress, status, 0, len(selected_classes)
            )
            merged = merge_same_class_annotations(annotations, predictions)
            progress.progress(1.0, text="Integrating BBoxes")
            st.session_state["annotations"] = merged
            annotations = merged
            if save_current_draft(dataset_dir, image_path, image, annotations):
                st.session_state["flash_message"] = f"候補BBoxを作成しました: {len(predictions)}件（要確認）"
                st.rerun()

    if batch_unprocessed_button:
        if not checkpoint.is_file():
            st.error(f"Checkpoint not found: {checkpoint}")
        elif not selected_names:
            st.error("Select at least one class before batch annotation.")
        else:
            selected_classes = [get_class_by_name(classes, name) for name in selected_names]
            selected_class_names = {item["name"] for item in selected_classes}
            pending_images = []
            for path in images:
                existing = load_draft(dataset_dir, path.name)
                status_value = existing.get("status") if existing else "unprocessed"
                processed_classes = processed_classes_from_draft(existing)
                if status_value == "approved" and not overwrite_batch:
                    continue
                if existing is not None and selected_class_names.issubset(processed_classes) and not overwrite_batch:
                    continue
                pending_images.append(path)
            if not pending_images:
                st.info("未処理の画像はありません（対象クラスはすべて処理済みです）。")
            else:
                status = st.empty()
                progress = st.progress(0, text="Preparing SAM 3.1 model for batch annotation...")
                predictor = get_predictor(str(checkpoint))
                completed_images = 0
                skipped_images = 0
                failures: list[str] = []
                for image_number, batch_image_path in enumerate(pending_images, start=1):
                    try:
                        existing = load_draft(dataset_dir, batch_image_path.name)
                        if existing is not None and existing.get("status") == "approved" and not overwrite_batch:
                            skipped_images += 1
                            continue
                        batch_image = Image.open(batch_image_path).convert("RGB")
                        status.write(f"Image {image_number} / {len(pending_images)}: {batch_image_path.name}")
                        batch_annotations = run_prompt_for_image(
                            predictor, batch_image,
                            {"mode": "Text", "classes": selected_classes}, confidence,
                        )
                        base_annotations = annotations_from_draft(existing)
                        merged = merge_same_class_annotations(base_annotations, batch_annotations)
                        processed_classes = processed_classes_from_draft(existing) | selected_class_names
                        save_draft(
                            dataset_dir, batch_image_path.name, batch_image.width, batch_image.height, merged,
                            prompt={"mode": "Text", "classes": sorted(processed_classes)}, status="needs_review",
                        )
                        completed_images += 1
                        progress.progress(image_number / len(pending_images), text=f"Image {image_number} / {len(pending_images)} complete")
                    except Exception as error:
                        failures.append(f"{batch_image_path.name}: {error}")
                summary = f"完了: {completed_images} 件 / スキップ: {skipped_images} 件 / 失敗: {len(failures)} 件"
                if failures:
                    st.warning(summary)
                    st.code("\n".join(failures))
                else:
                    st.session_state["loaded_image"] = None
                    st.session_state["flash_message"] = summary + "（すべて下書き保存・要確認）"
                    st.rerun()

    # streamlit-drawable-canvas (pinned 0.9.3) only correctly loads the
    # background image for one st_canvas() call per script run; mounting
    # several simultaneously leaves all but one blank. Only one canvas tool
    # is therefore ever rendered at a time.
    selected_annotation_id = st.session_state.get("selected_annotation_id")
    pending_active_tool = st.session_state.pop("pending_active_tool", None)
    if pending_active_tool:
        st.session_state["active_tool"] = pending_active_tool
    tool_expanded = selected_annotation_id is not None or st.session_state.get("active_tool", "非表示") != "非表示"
    with st.expander("修正ツール（誤りがある画像だけ使用）", expanded=tool_expanded):
        st.caption("不足はPoint / Boxで追加し、位置やクラスの誤りはBBox一覧の「修正」から直せます。")
        active_tool = st.radio(
            "操作", ["非表示", "Point / Box で指定", "選択したBBoxを編集", "手動でBBoxを追加"],
            horizontal=True, key="active_tool",
            format_func=lambda value: {
                "非表示": "閉じる",
                "Point / Box で指定": "不足をSAMで追加",
                "選択したBBoxを編集": "選択中BBoxを編集",
                "手動でBBoxを追加": "手動で追加",
            }[value],
        )

    # A BBox selection is only relevant while its edit tool is active.  Clear
    # it when the user switches to Point/Box or manual addition so that stale
    # class/confidence controls and selection highlighting do not remain on
    # screen alongside the newly selected workflow.
    if active_tool != "選択したBBoxを編集" and selected_annotation_id is not None:
        st.session_state["selected_annotation_id"] = None
        selected_annotation_id = None

    if active_tool == "Point / Box で指定":
        render_sam_add_tool(
            image_path,
            image,
            dataset_dir,
            checkpoint,
            classes,
            selected_names,
            available_names,
            confidence,
            image_key,
            annotations,
        )

    if queue_prompt_button:
        if advanced_mode == "Point" and not st.session_state.get("prompt_points"):
            st.error("Add at least one point before queueing this image.")
        elif advanced_mode == "Box" and not st.session_state.get("prompt_box"):
            st.error("Draw a box before queueing this image.")
        elif not target_name:
            st.error("Select a Target class before queueing this image.")
        else:
            target_config = get_class_by_name(classes, target_name)
            queue = st.session_state.setdefault("prompt_queue", {})
            queue[str(image_path)] = {
                "mode": advanced_mode,
                "class": target_config,
                "points": st.session_state.get("prompt_points", []),
                "labels": st.session_state.get("prompt_point_labels", []),
                "box": st.session_state.get("prompt_box"),
                "confidence": confidence,
            }
            st.success(f"Queued {image_path.name}. Queued images: {len(queue)}")

    if run_queue_button:
        queue = st.session_state.get("prompt_queue", {})
        if not queue:
            st.info("No queued Point/Box prompts. Queue a prompt for each image first.")
        else:
            status = st.empty()
            progress = st.progress(0, text="Preparing SAM 3.1 model for queued prompts...")
            predictor = get_predictor(str(checkpoint))
            completed = 0
            skipped = 0
            failures: list[str] = []
            failed_queue: dict[str, dict] = {}
            for index, (queued_path, prompt_spec) in enumerate(queue.items(), start=1):
                try:
                    queued_image_path = Path(queued_path)
                    existing = load_draft(dataset_dir, queued_image_path.name)
                    if existing is not None and existing.get("status") == "approved" and not overwrite_batch:
                        skipped += 1
                        continue
                    queued_image = Image.open(queued_image_path).convert("RGB")
                    status.write(f"Image {index} / {len(queue)}: {queued_image_path.name}")
                    queued_annotations = run_prompt_for_image(predictor, queued_image, prompt_spec, prompt_spec["confidence"])
                    base_annotations = annotations_from_draft(existing)
                    merged = merge_same_class_annotations(base_annotations, queued_annotations)
                    save_draft(
                        dataset_dir, queued_image_path.name, queued_image.width, queued_image.height, merged,
                        prompt={"mode": prompt_spec["mode"]}, status="needs_review",
                    )
                    completed += 1
                    progress.progress(index / len(queue), text=f"Queued image {index} / {len(queue)} complete")
                except Exception as error:
                    failures.append(f"{queued_path}: {error}")
                    failed_queue[queued_path] = prompt_spec
            # Completed and skipped entries should not run again accidentally.
            # Keep only failures so the user can retry them after correcting the
            # underlying problem.
            st.session_state["prompt_queue"] = failed_queue
            summary = f"完了: {completed} 件 / スキップ: {skipped} 件 / 失敗: {len(failures)} 件"
            if failures:
                st.warning(summary)
                st.code("\n".join(failures))
            else:
                st.session_state["loaded_image"] = None
                st.session_state["flash_message"] = summary + "（すべて下書き保存・要確認）"
                st.rerun()

    selected_annotation_id = st.session_state.get("selected_annotation_id")
    left, right = st.columns([2.6, 1.4])
    with left:
        st.subheader("画像とBBox")
        selected_display_index = next(
            (index for index, item in enumerate(annotations) if item["annotation_id"] == selected_annotation_id),
            None,
        )
        st.image(draw_annotations(image, annotations, selected_display_index), caption=image_path.name, use_column_width=True)

    with right:
        st.subheader(f"検出結果（{len(annotations)}件）")
        st.caption("問題なければ修正せず、そのまま確定できます。")
        if annotations:
            for index, annotation in enumerate(annotations):
                display_name = get_class_by_name(classes, annotation["class_name"])["display_name"]
                row, select_column, delete_column = st.columns([3.2, 1, 1])
                verified = "✓" if annotation.get("human_verified") else "・未確認"
                row.write(f"#{index + 1} {display_name} {float(annotation['confidence']):.2f} {verified}")
                if select_column.button("修正", key=f"select_{annotation['annotation_id']}"):
                    st.session_state["selected_annotation_id"] = annotation["annotation_id"]
                    st.session_state["pending_active_tool"] = "選択したBBoxを編集"
                    bump_canvas_generation()
                    st.rerun()
                if delete_column.button("除外", key=f"delete_{annotation['annotation_id']}"):
                    push_undo(image_key, annotations)
                    annotations.pop(index)
                    if st.session_state.get("selected_annotation_id") == annotation["annotation_id"]:
                        st.session_state["selected_annotation_id"] = None
                    save_current_draft(dataset_dir, image_path, image, annotations)
                    st.rerun()
        else:
            st.info("候補はまだありません。「候補を自動作成」を押してください。対象物がない画像は、そのまま対象なしで確定できます。")

        st.divider()
        if st.button("↶ 直前の修正を元に戻す"):
            restored = pop_undo(image_key)
            if restored is None:
                st.info("これ以上元に戻す操作はありません。")
            else:
                st.session_state["annotations"] = restored
                st.session_state["selected_annotation_id"] = None
                save_current_draft(dataset_dir, image_path, image, restored)
                st.rerun()

        selected = next((item for item in annotations if item["annotation_id"] == selected_annotation_id), None)
        if selected is not None:
            st.subheader("選択したBBoxを編集")
            selected_name = st.selectbox(
                "Class", available_names,
                index=available_names.index(selected["class_name"]) if selected["class_name"] in available_names else 0,
                key=f"edit_class_{selected['annotation_id']}",
                format_func=lambda name: get_class_by_name(classes, name)["display_name"],
            )
            selected_config = get_class_by_name(classes, selected_name)
            if selected["class_name"] != selected_config["name"] or selected["class_id"] != selected_config["id"]:
                selected["class_name"] = selected_config["name"]
                selected["class_id"] = selected_config["id"]
                if save_current_draft(dataset_dir, image_path, image, annotations):
                    st.rerun()
            new_confidence = st.number_input(
                "Confidence", 0.0, 1.0, float(selected.get("confidence", 1.0)), 0.01,
                key=f"edit_conf_{selected['annotation_id']}",
            )
            if new_confidence != selected.get("confidence"):
                selected["confidence"] = new_confidence
                if save_current_draft(dataset_dir, image_path, image, annotations):
                    st.rerun()

    # Geometry editing / manual add use the same (proven) display size as the
    # main result image. Only one of these mounts per rerun (see active_tool
    # above): mounting several st_canvas() at once leaves all but one blank.
    if active_tool == "選択したBBoxを編集":
        if selected is None:
            st.info("先にBBox一覧から「編集」を選んでください。")
        else:
            st.subheader("選択したBBoxの位置・大きさをドラッグで編集")
            edit_display_size = display_size(image)
            edit_image = image.resize(edit_display_size)
            edit_box_display = original_to_display_box(selected["bbox_xyxy"], image.size, edit_display_size)
            edit_canvas = st_canvas(
                background_image=edit_image,
                initial_drawing={
                    "version": "4.4.0",
                    "objects": [{
                        "type": "rect",
                        "left": edit_box_display[0], "top": edit_box_display[1],
                        "width": edit_box_display[2] - edit_box_display[0],
                        "height": edit_box_display[3] - edit_box_display[1],
                        "fill": "rgba(0,0,0,0)", "stroke": "#ff4628", "strokeWidth": 3,
                    }],
                },
                drawing_mode="transform",
                stroke_color="#ff4628",
                update_streamlit=True,
                width=edit_display_size[0],
                height=edit_display_size[1],
                key=f"edit_canvas_{image_path.stem}_{selected['annotation_id']}_{st.session_state.get('edit_generation', 0)}",
            )
            if st.button("この位置・大きさを適用"):
                rectangles = [item for item in (edit_canvas.json_data or {}).get("objects", []) if item.get("type") == "rect"]
                if not rectangles:
                    st.error("編集用のBBoxが見つかりません。")
                else:
                    item = rectangles[-1]
                    x1 = float(item.get("left", 0))
                    y1 = float(item.get("top", 0))
                    x2 = x1 + float(item.get("width", 0)) * float(item.get("scaleX", 1))
                    y2 = y1 + float(item.get("height", 0)) * float(item.get("scaleY", 1))
                    new_box = display_to_original_box([x1, y1, x2, y2], image.size, edit_display_size)
                    new_box = [
                        max(0.0, min(new_box[0], new_box[2])), max(0.0, min(new_box[1], new_box[3])),
                        min(float(image.width), max(new_box[0], new_box[2])), min(float(image.height), max(new_box[1], new_box[3])),
                    ]
                    if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
                        st.error("BBoxの幅・高さは正の値にしてください。")
                    else:
                        push_undo(image_key, annotations)
                        selected["bbox_xyxy"] = new_box
                        st.session_state["edit_generation"] = st.session_state.get("edit_generation", 0) + 1
                        save_current_draft(dataset_dir, image_path, image, annotations)
                        st.rerun()

    if active_tool == "手動でBBoxを追加":
        st.subheader("手動でBBoxを追加（SAM推論なし）")
        manual_display_size = display_size(image)
        manual_image = image.resize(manual_display_size)
        manual_canvas = st_canvas(
            background_image=manual_image,
            drawing_mode="rect",
            stroke_color="#38bdf8",
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            update_streamlit=True,
            width=manual_display_size[0],
            height=manual_display_size[1],
            key=f"manual_canvas_{image_path.stem}_{st.session_state.get('manual_generation', 0)}",
        )
        manual_class = st.selectbox(
            "手動追加のClass", available_names, key="manual_add_class",
            format_func=lambda name: get_class_by_name(classes, name)["display_name"],
        ) if available_names else None
        if st.button("この矩形をBBoxとして追加"):
            rectangles = [item for item in (manual_canvas.json_data or {}).get("objects", []) if item.get("type") == "rect"]
            if not rectangles or not manual_class:
                st.error("矩形を描画し、Classを選んでください。")
            else:
                item = rectangles[-1]
                x1 = float(item.get("left", 0))
                y1 = float(item.get("top", 0))
                x2 = x1 + float(item.get("width", 0)) * float(item.get("scaleX", 1))
                y2 = y1 + float(item.get("height", 0)) * float(item.get("scaleY", 1))
                new_box = display_to_original_box([x1, y1, x2, y2], image.size, manual_display_size)
                if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
                    st.error("BBoxの幅・高さは正の値にしてください。")
                else:
                    manual_config = get_class_by_name(classes, manual_class)
                    push_undo(image_key, annotations)
                    annotations.append({
                        "annotation_id": uuid.uuid4().hex,
                        "class_id": manual_config["id"],
                        "class_name": manual_config["name"],
                        "bbox_xyxy": new_box,
                        "confidence": 1.0,
                        "source": "manual",
                        "human_verified": False,
                    })
                    st.session_state["manual_generation"] = st.session_state.get("manual_generation", 0) + 1
                    save_current_draft(dataset_dir, image_path, image, annotations)
                    st.rerun()

    st.divider()
    st.subheader("確認結果")
    nav_prev, nav_later, nav_confirm = st.columns([1, 1.2, 2])
    if nav_prev.button("← 前へ", disabled=current_index == 0):
        st.session_state["image_index"] = current_index - 1
        st.rerun()

    next_index = next_pending_index(images, dataset_dir, current_index)
    if nav_later.button("今回は保留", disabled=next_index is None, help="下書きを残して次の未確認画像へ進みます"):
        if image_status == "approved" or save_current_draft(dataset_dir, image_path, image, annotations):
            st.session_state["image_index"] = next_index
            st.rerun()

    def approve_and_advance(final_annotations: list[dict]) -> None:
        errors = validate_annotations(final_annotations, image.width, image.height, classes)
        if errors:
            for error in errors:
                st.error(error)
            return
        for annotation in final_annotations:
            annotation["human_verified"] = True
        try:
            save_annotations(dataset_dir, image_path, image.width, image.height, final_annotations, classes)
        except Exception as error:
            st.error(f"確定保存に失敗しました。画像は移動しません: {error}")
            return
        if not save_current_draft(dataset_dir, image_path, image, final_annotations):
            return
        try:
            save_draft(dataset_dir, image_path.name, image.width, image.height, final_annotations, status="approved")
        except OSError as error:
            st.error(f"確定状態の保存に失敗しました: {error}")
            return
        st.session_state["annotations"] = final_annotations
        st.session_state["approved_revision"] = st.session_state.get("approved_revision", 0) + 1
        st.session_state["flash_message"] = f"{image_path.name} を確定しました（BBox {len(final_annotations)}件）。"
        next_index_after_approval = next_pending_index(images, dataset_dir, current_index)
        if next_index_after_approval is not None:
            st.session_state["image_index"] = next_index_after_approval
            st.rerun()
        st.rerun()

    if image_status == "approved":
        next_unapproved = next_pending_index(images, dataset_dir, current_index)
        if nav_confirm.button("次の未確認画像へ →", type="primary", disabled=next_unapproved is None):
            st.session_state["image_index"] = next_unapproved
            st.rerun()
        if next_unapproved is None:
            nav_confirm.success("すべて確認済みです")
    elif annotations:
        if nav_confirm.button("確定して次へ", type="primary"):
            approve_and_advance(annotations)
    else:
        no_target_confirmed = nav_confirm.checkbox("対象物なしを確認しました", key="no_target_checkbox")
        if nav_confirm.button("対象なしで確定して次へ", type="primary", disabled=not no_target_confirmed):
            approve_and_advance([])


if __name__ == "__main__":
    main()

