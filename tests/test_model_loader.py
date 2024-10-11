import pytest
from src.model_loader import load_model

def test_load_model():
    model = load_model('mobilenet_v2')
    assert model is not None, "Model should not be None"

def test_load_model_invalid():
    with pytest.raises(ValueError):
        load_model('invalid_model')
