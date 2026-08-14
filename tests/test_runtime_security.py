from io import BytesIO

import pytest
import torch
from PIL import Image, UnidentifiedImageError

import aiml_notebooks


def test_torch_valid_and_invalid_shapes() -> None:
    model = torch.nn.Linear(2, 1)

    assert model(torch.tensor([[1.0, 2.0]])).shape == (1, 1)
    with pytest.raises(RuntimeError):
        model(torch.ones(1, 3))


def test_pillow_rejects_invalid_image_payload() -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"not-an-image")).load()


def test_shared_library_imports() -> None:
    assert aiml_notebooks.__name__ == "aiml_notebooks"
