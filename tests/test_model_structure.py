"""Tests for model architecture properties."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import pytest


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_model_output_matches_num_classes():
    """Model output layer units must equal num_classes argument."""
    from indoor_outdoor.modeling.model import create_model

    model = create_model(num_classes=1, image_size=224)
    assert model.output_shape[-1] == 1


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_model_accepts_configured_input_size():
    """Model input shape must match the image_size passed to create_model."""
    from indoor_outdoor.modeling.model import create_model

    model = create_model(num_classes=2, image_size=224)
    assert model.input_shape == (None, 224, 224, 3)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_model_is_not_trainable_base():
    """ResNet base model must be frozen (trainable=False) for transfer learning."""
    from indoor_outdoor.modeling.model import create_model

    model = create_model(num_classes=2, image_size=224)
    # The ResNet base is the layer named 'resnet50v2'
    base_layers = [layer for layer in model.layers if "resnet" in layer.name.lower()]
    assert base_layers, "ResNet base layer not found in model"
    assert not base_layers[0].trainable
