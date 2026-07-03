@echo off
cd /d "%~dp0\.."
echo Starting Keiba AI race-day timed LINE watcher...
echo.
echo This will refresh odds/tickets and send LINE notices around:
echo   - 08:00 morning race list
echo   - 5 minutes before each race
echo   - 3 minutes before each race
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\watch_race_day_timed_line_alerts.ps1" -SendIfConfigured -IntervalSeconds 60
echo.
echo Watcher stopped.
pause
