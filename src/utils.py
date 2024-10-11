import json
import logging


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def setup_logging(log_level, log_file):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",
    )
    logging.info(
        "Logging setup complete"
    )  # Add a log message to ensure the file is created
