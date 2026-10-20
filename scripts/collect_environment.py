import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from common import ROOT, RUNS, setup

setup()
report = {"python": sys.version, "platform": platform.platform(), "project": str(ROOT)}
for package in ("jax", "jaxlib", "flax", "torch", "numpy", "lerobot"):
    try:
        report[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report[package] = "not installed"
for command in (["nvidia-smi"], ["df", "-h", str(ROOT)]):
    if shutil.which(command[0]):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            report[command[0]] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as error:
            report[command[0]] = str(error)
try:
    import jax
    report["jax_devices"] = [str(device) for device in jax.devices()]
except Exception as error:
    report["jax_devices_error"] = repr(error)
target = RUNS / "reports/environment.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)
