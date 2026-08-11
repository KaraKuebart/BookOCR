"""Tests for the preprocessing step of the pipeline."""

import numpy as np
from PIL import Image

from src import yolo_segment_predict


def test_downscale_image_halves_the_resolution():
    """A scale factor of two halves width and height."""
    image = np.zeros((100, 60, 3), dtype=np.uint8)

    scaled = yolo_segment_predict.downscale_image(image, 2)

    assert scaled.shape[:2] == (50, 30)


def test_normalize_image_stretches_contrast(tmp_path, monkeypatch):
    """A low contrast image is stretched to the full value range."""
    monkeypatch.setattr(yolo_segment_predict, "input_folder", str(tmp_path))
    input_path = tmp_path / "page.png"
    output_path = tmp_path / "page_norm.png"
    low_contrast = np.full((40, 20, 3), 100, dtype=np.uint8)
    low_contrast[:20] = 140
    Image.fromarray(low_contrast).save(input_path)

    yolo_segment_predict.normalize_image(
        str(input_path), str(output_path), downscale_factor=2
    )

    result = np.array(Image.open(output_path))
    assert result.shape[:2] == (20, 10)
    assert result.min() == 0
    assert result.max() == 255


def test_preprocess_writes_one_normalized_file_per_input(
    tmp_path, monkeypatch, page_photo
):
    """Every input photo ends up in the ``norm`` subfolder."""
    monkeypatch.setattr(yolo_segment_predict, "input_folder", str(tmp_path))
    Image.open(page_photo).save(tmp_path / "page.jpg")

    yolo_segment_predict.preprocess(downscale_factor=4)

    normalized = list((tmp_path / "norm").glob("*.jpg"))
    assert [path.name for path in normalized] == ["page.jpg"]
