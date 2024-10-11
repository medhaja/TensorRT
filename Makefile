# Define the Python interpreter path for Python 3.10.11
PYTHON_3_10_11 = "C:/Program Files/Python310/python.exe"

# Define the source directory and configuration file
SRC_DIR = src
CONFIG_FILE = config/config.json

# Define the path to flake8
FLAKE8 = "C:/Users/medha/AppData/Roaming/Python/Python310/Scripts/flake8"

# Define the virtual environment directory
VENV_DIR = venv

# Define the requirements file
REQUIREMENTS_FILE = requirements.txt

# Define the linting and testing targets
.PHONY: lint test run clean venv install style all

lint:
	@echo "Linting code..."
	$(FLAKE8) $(SRC_DIR)

test:
	@echo "Running tests..."
	pytest

run:
	@echo "Running the script..."
	$(PYTHON_3_10_11) $(SRC_DIR)/main.py --config $(CONFIG_FILE)

clean:
	@echo "Running cleanup script..."
	clean.bat

venv:
	@echo "Creating virtual environment with Python 3.10.11..."
	$(PYTHON_3_10_11) -m venv $(VENV_DIR)

install:
	@echo "Installing requirements..."
	@echo "Running install_requirements.bat script..."
	@echo "Script location: $(shell pwd)/install_requirements.bat"
	install_requirements.bat

style:
	@echo "Applying code styling..."
	style.bat

all: lint test run