@echo off
REM PathSnap demo - LAPTOP 2 (Evaluator UI). Double-click to run.
REM Asks for Laptop 1's IP, tests the connection and opens the UI already connected.
REM   LAPTOP2_START_DEMO.bat 192.168.43.152   -> skip the question
title PathSnap - LAPTOP 2 (UI)
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo\laptop2_start.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo\laptop2_start.ps1" -Api %1
)
echo.
pause
