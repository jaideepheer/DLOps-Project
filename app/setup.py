# See: https://stackoverflow.com/a/63096701/10027894

import sys
import subprocess
import pkg_resources

required = {
    "streamlit",
    "Pillow",
    "pandas",
    "numpy"
}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-U"])

print("Setup done.")

# Run streamlit
print(f"Running bootstrap {sys.argv[1:]}")
subprocess.check_call([sys.executable, *sys.argv[1:]])
