# Inference Project

This project compares the inference performance of PyTorch and TensorRT on a set of images using pre-trained models.

## Prerequisites
- Python 3.x
- PyTorch
- ONNX
- TensorRT
- PyCUDA
- Pillow
- argparse
- logging
- numpy
- matplotlib

## Installation
```bash
pip install -r requirements.txt
# PyTorch and TensorRT Inference Comparison
```
This project compares the inference performance of a PyTorch model and its TensorRT-optimized counterpart. It includes functionalities to load a model, export it to ONNX format, convert the ONNX model to TensorRT, and run inference on a set of images. The results are logged, and a performance report is generated, including mean inference times and throughput comparisons.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Install TensorRT:**

    Follow the official [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT on your system.

## Usage

To run the inference comparison, use the following command:

```bash
python main.py --config path/to/config.json
```
Replace path/to/config.json with the path to your configuration file.
Configuration

The configuration file should be a JSON file with the following structure:

{
    "model": "path/to/your/model.pth",
    "image_folder": "path/to/your/image/folder",
    "output_dir": "path/to/output/directory",
    "log_level": "INFO"
}

    model: Path to the PyTorch model file.
    image_folder: Path to the folder containing images for inference.
    output_dir: Path to the directory where the performance report and plot will be saved.
    log_level: Logging level (e.g., INFO, DEBUG, WARNING, ERROR).

Output

The script generates the following outputs:

    Log File:
        logs/inference.log: Contains detailed logs of the inference process.

    Performance Report:
        output_dir/performance_report.json: JSON file containing mean inference times and throughput for both PyTorch and TensorRT.

    Performance Comparison Plot:
        output_dir/performance_comparison.png: A plot comparing the mean inference times and throughput of PyTorch and TensorRT.

Dependencies

The project requires the following dependencies:

    argparse
    json
    logging
    os
    sys
    matplotlib
    numpy
    onnx
    torch
    tensorrt

These dependencies can be installed using the requirements.txt file:

pip install -r requirements.txt

License

This project is licensed under the MIT License. See the LICENSE file for details.
