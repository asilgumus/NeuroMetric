"""Stage only public frontend files: no model weights, source MRI or credentials."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "dist-demo"

if __name__ == "__main__":
    DESTINATION.mkdir(exist_ok=True)
    for name in ("index.html", "report.html", "resources.html", "longitudinal.html", "styles.css", "app.js", "DATA_SOURCES.md"):
        shutil.copy2(ROOT / name, DESTINATION / name)
    shutil.copytree(ROOT / "assets", DESTINATION / "assets", dirs_exist_ok=True)
    print(f"Static hosting files staged at {DESTINATION}; no remote deployment performed")
