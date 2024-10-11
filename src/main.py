import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch

import tensorrt as trt

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.image_loader import load_images_from_folder
from src.inference import run_pytorch_inference, run_tensorrt_inference
from src.model_loader import load_model
from src.utils import load_config, setup_logging


def setup_logging(log_level, log_file):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set the log level for handlers
    file_handler.setLevel(log_level)
    console_handler.setLevel(log_level)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main(config):
    # Set up logging
    setup_logging(config["log_level"], "logs/inference.log")

    # Load the model
    logging.info(f"Loading the model: {config['model']}")
    model = load_model(config["model"])

    # Load images from the specified folder
    logging.info(f"Loading images from folder: {config['image_folder']}")
    images = load_images_from_folder(config["image_folder"])

    # Create a dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX format
    onnx_model_path = f"{config['model']}.onnx"
    logging.info(f"Exporting the model to ONNX format: {onnx_model_path}")
    torch.onnx.export(model, dummy_input, onnx_model_path)

    # Load the ONNX model
    logging.info(f"Loading the ONNX model: {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model loaded and checked successfully.")

    # Create a TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create a TensorRT builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # Create a builder config
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # Parse the ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_model.SerializeToString()):
        for error in range(parser.num_errors):
            logging.error(parser.get_error(error))
    logging.info("ONNX model parsed successfully.")

    # Build the TensorRT engine
    serialized_engine = builder.build_serialized_network(network, builder_config)
    logging.info("TensorRT engine built successfully.")

    # Save the engine to a file
    trt_model_path = f"{config['model']}.trt"
    with open(trt_model_path, "wb") as f:
        f.write(serialized_engine)
    logging.info(f"TensorRT engine saved to: {trt_model_path}")

    # Run inference for all images with PyTorch model
    pytorch_times = []
    for image in images:
        pytorch_output, pytorch_time = run_pytorch_inference(model, image)
        pytorch_times.append(pytorch_time)
    logging.info("PyTorch inference completed for all images.")

    # Load the TensorRT engine
    with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    logging.info("TensorRT engine loaded successfully.")

    # Run inference for all images with TensorRT engine
    tensorrt_times = []
    for image in images:
        tensorrt_output, tensorrt_time = run_tensorrt_inference(engine, image)
        tensorrt_times.append(tensorrt_time)
    logging.info("TensorRT inference completed for all images.")

    # Calculate the mean inference times
    pytorch_mean_time = np.mean(pytorch_times)
    tensorrt_mean_time = np.mean(tensorrt_times)

    # Calculate throughput (images per second)
    pytorch_throughput = 1 / pytorch_mean_time
    tensorrt_throughput = 1 / tensorrt_mean_time

    # Calculate speedup factor
    speedup_factor = pytorch_mean_time / tensorrt_mean_time

    # Print the results
    logging.info(f"PyTorch Mean Inference Time: {pytorch_mean_time:.4f} seconds")
    logging.info(f"TensorRT Mean Inference Time: {tensorrt_mean_time:.4f} seconds")
    logging.info(f"PyTorch Throughput: {pytorch_throughput:.2f} images/second")
    logging.info(f"TensorRT Throughput: {tensorrt_throughput:.2f} images/second")
    logging.info(f"TensorRT is {speedup_factor:.2f} times faster than PyTorch")

    # Compare the outputs (optional)
    logging.info("Output comparison (first 10 values):")
    logging.info(f"PyTorch: {pytorch_output[0][:10]}")
    logging.info(f"TensorRT: {tensorrt_output[0][:10]}")

    # Save performance report
    performance_report = {
        "pytorch_mean_time": pytorch_mean_time,
        "tensorrt_mean_time": tensorrt_mean_time,
        "pytorch_throughput": pytorch_throughput,
        "tensorrt_throughput": tensorrt_throughput,
    }
    with open(os.path.join(config["output_dir"], "performance_report.json"), "w") as f:
        json.dump(performance_report, f)
    logging.info(
        f"Performance report saved to: {os.path.join(config['output_dir'], 'performance_report.json')}"
    )

    # Plot the results in a single figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Mean Inference Time Comparison
    axs[0].bar(
        ["PyTorch", "TensorRT"],
        [pytorch_mean_time, tensorrt_mean_time],
        color=["blue", "green"],
    )
    axs[0].set_xlabel("Framework")
    axs[0].set_ylabel("Mean Inference Time (seconds)")
    axs[0].set_title("Mean Inference Time Comparison")

    # Throughput Comparison
    axs[1].bar(
        ["PyTorch", "TensorRT"],
        [pytorch_throughput, tensorrt_throughput],
        color=["blue", "green"],
    )
    axs[1].set_xlabel("Framework")
    axs[1].set_ylabel("Throughput (images/second)")
    axs[1].set_title("Throughput Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "performance_comparison.png"))
    plt.show()
    logging.info(
        f"Performance comparison plot saved to: {os.path.join(config['output_dir'], 'performance_comparison.png')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on images using PyTorch and TensorRT"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.config)

    main(config)
