import time

import numpy as np
import pycuda.autoinit  # Automatically initialize the CUDA context
import pycuda.driver as cuda
import torch

import tensorrt as trt


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

    # Convert input_data to numpy array
    input_data_np = input_data.numpy()

    cuda.memcpy_htod_async(d_input, input_data_np, stream)

    start_time = time.time()
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
    )
    stream.synchronize()
    end_time = time.time()

    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    return output_data, end_time - start_time
