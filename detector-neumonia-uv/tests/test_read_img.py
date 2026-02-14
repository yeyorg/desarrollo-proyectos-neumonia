import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.read_img import ImageLoader

def test_init_with_dicom():
    """Test que ImageLoader se inicializa correctamente con un archivo DICOM."""
    path = "dummy.dcm"
    mock_img = MagicMock()
    mock_img.pixel_array = np.full((100, 100), 100, dtype=np.uint8)
    
    with patch("pydicom.dcmread", return_value=mock_img):
        loader = ImageLoader(path)
        assert loader.img == mock_img
        assert loader.img2show is not None
        assert loader.img_RGB is not None

def test_get_img_RGB():
    """Test que get_img_RGB retorna la imagen normalizada."""
    path = "test.dcm"
    mock_img = MagicMock()
    mock_img.pixel_array = np.ones((10, 10), dtype=np.uint8) * 100
    
    with patch("pydicom.dcmread", return_value=mock_img):
        loader = ImageLoader(path)
        img_rgb = loader.get_img_RGB()
        assert img_rgb.shape == (10, 10, 3)
        assert img_rgb.dtype == np.uint8


