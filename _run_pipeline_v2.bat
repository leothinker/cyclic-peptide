@echo off
REM Full re-extract with the new _ocr_cell CH-stripped-borders fix.
setlocal
cd /d D:\GitHub\drugdesign\cyclic-peptide
set TS=20261015_171500
set LOG=data\processed\pipeline_v2_%TS%.log
set DONE=data\processed\.pipeline_done_%TS%
echo log=%LOG% > data\processed\.pipeline_cmd.txt
echo done=%DONE% >> data\processed\.pipeline_cmd.txt
.venv\Scripts\python.exe -u main.py --rec-top-k 5 > "%LOG%" 2>&1
echo done > "%DONE%"
echo finished with code %ERRORLEVEL% >> "%LOG%"
