import pkg_resources
import subprocess
import sys

# Đường dẫn đến file requirements.txt
requirements_file = "requirements.txt"

try:
    with open(requirements_file, "r") as f:
        dependencies = f.read().splitlines()

    print("=== Kiểm tra thư viện từ requirements.txt ===\n")
    for dependency in dependencies:
        try:
            pkg_resources.require(dependency)
            print(f"{dependency}: INSTALLED")
        except pkg_resources.DistributionNotFound:
            print(f"{dependency}: NOT INSTALLED")
        except pkg_resources.VersionConflict as e:
            print(f"{dependency}: VERSION CONFLICT ({e})")
except FileNotFoundError:
    print(f"File {requirements_file} không tồn tại!")
