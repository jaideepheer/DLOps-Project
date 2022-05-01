# See: https://stackoverflow.com/a/63096701/10027894

import sys
import subprocess
import pkg_resources

required = {
    "wandb",
    "rich",
}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-U"])

# install latest DALI
subprocess.check_call([sys.executable, "-m", "pip", "install", "--extra-index-url", "https://developer.download.nvidia.com/compute/redist", "-U", "nvidia-dali-cuda110"])