from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "autonomous vehicle" / "main.py"

if not TARGET.exists():
    raise FileNotFoundError(
        f"Could not find target script: {TARGET}\n"
        "Make sure the autonomous vehicle folder contains main.py."
    )

args = [sys.executable, str(TARGET)] + sys.argv[1:]
return_code = subprocess.call(args)
sys.exit(return_code)
