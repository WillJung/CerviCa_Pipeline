@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [CerviCa] scans_input 폴더 감시를 시작합니다.
echo 이 창을 닫으면 분석이 멈춥니다. 종료하려면 Ctrl+C 를 누르세요.
echo.

python watch_scans.py
if errorlevel 1 (
    echo.
    echo 실행에 실패했습니다. Python이 설치되어 있는지 확인하세요.
    pause
)
