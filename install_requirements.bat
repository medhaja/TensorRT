@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Deactivating virtual environment...
call venv\Scripts\deactivate
