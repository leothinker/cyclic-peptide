@echo off
REM Swap the freshly-pipelined compounds_combined.new.csv into the canonical
REM name once the user closes any editor that holds an exclusive lock on
REM compounds_combined.csv. Idempotent: if .new.csv is missing it does nothing.
setlocal
cd /d D:\GitHub\drugdesign\cyclic-peptide
if not exist "data\processed\compounds_combined.new.csv" (
    echo nothing to swap; missing data\processed\compounds_combined.new.csv
    exit /b 0
)
del /f /q "data\processed\compounds_combined.csv" >nul 2>nul
if exist "data\processed\compounds_combined.csv" (
    echo compounds_combined.csv still locked; close the editor and re-run.
    exit /b 1
)
move /y "data\processed\compounds_combined.new.csv" "data\processed\compounds_combined.csv"
echo swapped.
endlocal
