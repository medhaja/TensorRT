@echo off
echo Cleaning up...

REM Remove compiled Python files
del /s /q src\*.pyc
for /d /r src %%d in (__pycache__) do @rmdir /s /q "%%d"

REM Remove build artifacts
rmdir /s /q build dist
del /q *.egg-info

REM Remove log files
del /q logs\*.log

REM Remove temporary files
del /q temp\*

REM Remove pytest cache
rmdir /s /q .pytest_cache
