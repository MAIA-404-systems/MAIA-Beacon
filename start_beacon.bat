@echo off
title MAIA Beacon - GPU Worker Node
cd /d "%~dp0"

echo ===================================================
echo   MAIA Beacon - Demarrage du Worker GPU
echo ===================================================

if not exist ".env" (
    echo [!] Fichier .env non trouve. Creation depuis .env.example...
    copy ".env.example" ".env"
)

python app.py
pause
