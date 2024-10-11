@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Applying code styling...
black src
isort src

echo Deactivating virtual environment...
call venv\Scripts\deactivate
