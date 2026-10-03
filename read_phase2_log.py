from pathlib import Path

lines = Path("/tmp/phase2.log").read_text(errors="replace").splitlines()
for index, line in enumerate(lines):
    if any(token in line for token in ("Traceback", "AssertionError", "TypeError", "RuntimeError", "Error:")):
        print("\n".join(lines[index:index + 12]))
