import os
import argparse
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load a pre-trained MobileNetV2 model from PyTorch
def load_model(model_name):
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported")
    model.eval()
    return model

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            images.append(image)
    return images

# Function to run inference with PyTorch model
def run_pytorch_inference(model, input_data):
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    end_time = time.time()
    return output, end_time - start_time

# Function to run inference with TensorRT engine
def run_tensorrt_inference(engine, input_data):
    context = engine.create_execution_context()

    input_shape = (1, 3, 224, 224)
    output_shape = (1, 1000)
    d_input = cuda.mem_alloc(trt.volume(input_shape) * np.dtype(np.float32).itemsize)
    d_output = cuda.mem_alloc(trt.volume(output_shape) * np.dtype(np.float32).itemsize)

    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input_data.numpy(), stream)

    start_time = time.time()
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    stream.synchronize()
    end_time = time.time()

    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    return output_data, end_time - start_time

# Main function
def main(config):
    # Load the model
    model = load_model(config['model'])

    # Load images from the specified folder
    images = load_images_from_folder(config['image_folder'])

    # Create a dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX format
    onnx_model_path = f"{config['model']}.onnx"
    torch.onnx.export(model, dummy_input, onnx_model_path)

    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Create a TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create a TensorRT builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Create a builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB

    # Parse the ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_model.SerializeToString()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    # Build the TensorRT engine
    engine = builder.build_engine(network, config)

    # Save the engine to a file
    trt_model_path = f"{config['model']}.trt"
    with open(trt_model_path, "wb") as f:
        f.write(engine.serialize())

    # Run inference for all images with PyTorch model
    pytorch_times = []
    for image in images:
        pytorch_output, pytorch_time = run_pytorch_inference(model, image)
        pytorch_times.append(pytorch_time)

    # Load the TensorRT engine
    with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Run inference for all images with TensorRT engine
    tensorrt_times = []
    for image in images:
        tensorrt_output, tensorrt_time = run_tensorrt_inference(engine, image)
        tensorrt_times.append(tensorrt_time)

    # Calculate the mean inference times
    pytorch_mean_time = np.mean(pytorch_times)
    tensorrt_mean_time = np.mean(tensorrt_times)

    # Print the results
    logging.info(f"PyTorch Mean Inference Time: {pytorch_mean_time:.4f} seconds")
    logging.info(f"TensorRT Mean Inference Time: {tensorrt_mean_time:.4f} seconds")

    # Compare the outputs (optional)
    logging.info("Output comparison (first 10 values):")
    logging.info(f"PyTorch: {pytorch_output[0][:10]}")
    logging.info(f"TensorRT: {tensorrt_output[0][:10]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images using PyTorch and TensorRT")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)
