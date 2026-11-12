import pytest

tf = pytest.importorskip("tensorflow")

from mlkan.models.registry import build_model


@pytest.mark.parametrize("model_id", [f"M{i}" for i in range(8)] + ["M10", "M11"])
def test_2d_model_output_shape(model_id):
    model = build_model(model_id, (32, 32, 5), 2, {"base_channels": 8, "kan_widths": [4, 4, 8]})
    assert tuple(model(tf.zeros((2, 32, 32, 5))).shape) == (2, 32, 32, 1)


@pytest.mark.parametrize("model_id", [f"M{i}" for i in range(12)])
def test_1d_model_output_shape(model_id):
    model = build_model(model_id, (64, 4), 1, {"base_channels": 8, "kan_widths": [4, 4, 8]})
    assert tuple(model(tf.zeros((2, 64, 4))).shape) == (2, 64, 1)
