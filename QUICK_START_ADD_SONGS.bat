@echo off
echo ========================================
echo Opening Music Folder...
echo ========================================
echo.
echo This will open the music folder where you need to add your songs.
echo.
echo You should see 4 folders:
echo   - Happy
echo   - Sad  
echo   - Energetic
echo   - Calm
echo.
echo Just copy your music files (.mp3, .wav, .ogg) into the appropriate folder!
echo.
pause

cd /d "C:\Users\user\OneDrive\Desktop\github\AI-201\music"
explorer .

echo.
echo Folder opened! Add your songs and restart the app.
pause

