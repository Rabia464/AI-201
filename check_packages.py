"""Check which required packages are installed"""

required_packages = {
    'streamlit': 'streamlit',
    'opencv-python': 'cv2',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'pillow': 'PIL',
    'pygame': 'pygame'
}

installed = []
missing = []

print("Checking required packages...\n")

for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        installed.append(package_name)
        print(f"✓ {package_name} - INSTALLED")
    except ImportError:
        missing.append(package_name)
        print(f"✗ {package_name} - MISSING")

print("\n" + "="*50)
print(f"Installed: {len(installed)}/{len(required_packages)}")
print(f"Missing: {len(missing)}/{len(required_packages)}")

if missing:
    print("\nMissing packages:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\nTo install missing packages, run:")
    print("pip install -r requirements.txt")
else:
    print("\nAll packages are installed! ✓")

