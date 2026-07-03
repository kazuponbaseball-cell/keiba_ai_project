@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\run_live_dashboard_operation.ps1" -Loop -IntervalSeconds 60
pause
