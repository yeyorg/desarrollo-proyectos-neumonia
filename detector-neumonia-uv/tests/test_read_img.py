import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.read_img import ImageLoader

def test_read_dicom_file():
    path = "dummy.dcm"
    mock_img = MagicMock()
    mock_img.pixel_array = np.zeros((100, 100))
    
    with patch("pydicom.dcmread", return_value=mock_img) as mock_dcmread:
        result = ImageLoader.read_dicom_file(path)
        mock_dcmread.assert_called_once_with(path)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)

def test_read_jpg_file():
    path = "dummy.jpg"
    mock_array = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with patch("cv2.imread", return_value=mock_array) as mock_imread:
        result = ImageLoader.read_jpg_file(path)
        mock_imread.assert_called_once_with(path)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

def test_read_jpg_file_error():
    path = "invalid.jpg"
    with patch("cv2.imread", return_value=None):
        with pytest.raises(ValueError, match="No se pudo cargar la imagen"):
            ImageLoader.read_jpg_file(path)

def test_load_dicom():
    path = "test.dcm"
    with patch.object(ImageLoader, "read_dicom_file", return_value=np.zeros((10, 10))) as mock_read:
        result = ImageLoader.load(path)
        mock_read.assert_called_once_with(path)
        assert result.shape == (10, 10)

def test_load_jpg():
    path = "test.png"
    with patch.object(ImageLoader, "read_jpg_file", return_value=np.zeros((10, 10, 3))) as mock_read:
        result = ImageLoader.load(path)
        mock_read.assert_called_once_with(path)
        assert result.shape == (10, 10, 3)
