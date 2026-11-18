@echo off
REM PathSnap API launcher for Laptop 1 (run from anywhere; it switches to the repo folder).
REM Usage:  run_api.bat                 -> serves the evaluator_test trips
REM         run_api.bat student_test    -> serves another partition folder
REM Optional: set PATHSNAP_DATA=<folder containing the partition folders> before running.

cd /d "%~dp0"
if "%~1"=="" (set PATHSNAP_TRIPS_PARTITION=evaluator_test) else (set PATHSNAP_TRIPS_PARTITION=%~1)

echo === Pre-flight check of partition %PATHSNAP_TRIPS_PARTITION% ===
python scripts\check_trips.py %PATHSNAP_TRIPS_PARTITION%
if errorlevel 1 (
    echo.
    echo Pre-flight check reported problems ^(see above^). Fix them or press Ctrl+C to stop.
    pause
)

echo.
echo === This laptop's IP addresses (give one to Laptop 2 as http://IP:8000) ===
ipconfig | findstr /i "IPv4"
echo.
echo === Starting API on port 8000 (Ctrl+C to stop) ===
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
