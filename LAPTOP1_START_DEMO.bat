@echo off
REM PathSnap demo - LAPTOP 1 (Model + API). Double-click to run.
REM   LAPTOP1_START_DEMO.bat                 -> serves the evaluator_test trips
REM   LAPTOP1_START_DEMO.bat student_test    -> rehearse with the student-test trips
title PathSnap - LAPTOP 1 (API)
set PART=%~1
if "%PART%"=="" set PART=evaluator_test
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo\laptop1_start.ps1" -Partition %PART%
echo.
echo API stopped.
pause
