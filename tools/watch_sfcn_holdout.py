"""Monitor this exact evaluation job and retrieve results without launching work."""
from pathlib import Path
import subprocess
import time
import sys
import fcntl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import write_json


def main():
    output = ROOT / "artifacts/v3/heldout_contrast_cont4"
    output.mkdir(parents=True, exist_ok=True)
    ref = "asildoangm/brainage-sfcn-contrast-cont4-locked-held-out-test"
    with (output / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = subprocess.run(["/usr/bin/python", "-m", "kaggle", "kernels", "status", ref],
                                    capture_output=True, text=True, timeout=90)
            write_json(output / "status.json", {"ref": ref, "status": result.stdout.strip(),
                       "returncode": result.returncode, "error": result.stderr[-1000:]})
            print(result.stdout.strip(), flush=True)
            if result.returncode == 0 and ("COMPLETE" in result.stdout or "ERROR" in result.stdout or "CANCEL" in result.stdout):
                subprocess.run(["/usr/bin/python", "-m", "kaggle", "kernels", "output", ref,
                                "-p", str(output / "outputs"), "-q"], check=True, timeout=300)
                return
            time.sleep(180)


if __name__ == "__main__":
    main()
