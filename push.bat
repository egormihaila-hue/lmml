@echo off
cd /d "%~dp0"
echo === Uploading this folder to GitHub (lmml) ===
git init
git remote remove origin 2>nul
git remote add origin https://github.com/egormihaila-hue/lmml.git
git add .
git commit -m "snow solution"
git branch -M main
git pull origin main --allow-unrelated-histories --no-edit
git push -u origin main
echo.
echo === Done. Check https://github.com/egormihaila-hue/lmml ===
pause
