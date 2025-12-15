@echo off
echo ========================================
echo Fixing pip installation...
echo ========================================
echo.

cd /d "C:\Users\user\OneDrive\Desktop\github\AI-201"

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Step 2: Downloading get-pip.py...
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
if errorlevel 1 (
    echo Trying alternative method...
    python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
)

echo.
echo Step 3: Installing pip...
python get-pip.py

echo.
echo Step 4: Verifying pip installation...
python -m pip --version

echo.
echo Step 5: Installing required packages...
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Done! You can now run the app.
echo ========================================
del get-pip.py
pause

