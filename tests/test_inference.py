import torch
import numpy as np
import time
from src.inference import run_pytorch_inference, run_tensorrt_inference
from src.model_loader import load_model

def test_run_pytorch_inference():
    model = load_model('mobilenet_v2')
    input_data = torch.randn(1, 3, 224, 224)
    output, inference_time = run_pytorch_inference(model, input_data)
    assert output is not None, "Output should not be None"
    assert inference_time > 0, "Inference time should be positive"

def test_run_tensorrt_inference():
    # This test requires a TensorRT engine to be built and loaded
    # For the sake of this example, we'll mock the engine and input data
    class MockEngine:
        def create_execution_context(self):
            return self

        def execute_async_v2(self, bindings, stream_handle):
            time.sleep(0.1)  # Simulate a delay

    engine = MockEngine()
    input_data = torch.randn(1, 3, 224, 224)  # Use PyTorch tensor for consistency
    output, inference_time = run_tensorrt_inference(engine, input_data)
    assert output is not None, "Output should not be None"
    assert inference_time > 0, "Inference time should be positive"
