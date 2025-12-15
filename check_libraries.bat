@echo off
echo Checking installed libraries...
echo.

python -c "import streamlit; print('streamlit - OK')" 2>nul || echo streamlit - MISSING
python -c "import cv2; print('opencv-python - OK')" 2>nul || echo opencv-python - MISSING
python -c "import torch; print('torch - OK')" 2>nul || echo torch - MISSING
python -c "import torchvision; print('torchvision - OK')" 2>nul || echo torchvision - MISSING
python -c "import numpy; print('numpy - OK')" 2>nul || echo numpy - MISSING
python -c "import pandas; print('pandas - OK')" 2>nul || echo pandas - MISSING
python -c "import matplotlib; print('matplotlib - OK')" 2>nul || echo matplotlib - MISSING
python -c "import PIL; print('pillow - OK')" 2>nul || echo pillow - MISSING
python -c "import pygame; print('pygame - OK')" 2>nul || echo pygame - MISSING

echo.
echo To install missing packages, run: python -m pip install -r requirements.txt
echo Or try: py -m pip install -r requirements.txt
pause

