@echo off
cd /d "%~dp0"
echo ============================================================
echo  M10: XAI-Enhanced SwiGLU Psi-NN
echo  Web Application + XAI Dashboard
echo ============================================================
echo.
echo  Open browser at:  http://127.0.0.1:5000
echo  XAI Dashboard:    http://127.0.0.1:5000/xai
echo.
python webapp.py
pause