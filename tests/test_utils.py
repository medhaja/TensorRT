import os
import json
import logging  # Import the logging module
from src.utils import load_config, setup_logging

def test_load_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_data = {
        "model": "mobilenet_v2",
        "image_folder": "path/to/your/image/folder",
        "batch_size": 10,
        "log_level": "INFO",
        "output_dir": "results"
    }
    config_path.write_text(json.dumps(config_data))

    config = load_config(config_path)
    assert config == config_data, "Config should match the expected data"
