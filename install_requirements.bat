@echo off
cd /d "C:\Users\user\OneDrive\Desktop\github\AI-201"
echo Installing required libraries...
echo.

REM Try python -m pip first, if that fails try py -m pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Trying py -m pip...
    py -m pip install -r requirements.txt
)

echo.
echo Installation complete!
pause

