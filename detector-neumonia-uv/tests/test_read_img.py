import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.read_img import ImageLoader

def test_init_with_dicom():
    """
    Prueba que ImageLoader se inicializa correctamente con un archivo DICOM.
    Verifica que cuando se instancia un ImageLoader con una ruta de archivo DICOM,
    carga correctamente los datos de la imagen, crea una representación visualizable,
    y convierte la matriz de píxeles a formato RGB.
    Args:
        loader.img: contiene el objeto DICOM cargado
        loader.img2show: se completa con una representación de imagen visualizable
        loader.img_RGB: contiene los datos de imagen convertidos a RGB
    """
    
    path = "dummy.dcm"
    mock_img = MagicMock()
    mock_img.pixel_array = np.full((100, 100), 100, dtype=np.uint8)
    
    with patch("pydicom.dcmread", return_value=mock_img):
        loader = ImageLoader(path)
        assert loader.img == mock_img
        assert loader.img2show is not None
        assert loader.img_RGB is not None

def test_get_img_RGB():
    """
    Prueba que `get_img_RGB` convierte correctamente una imagen DICOM a formato RGB normalizado.
    Verifica que:
        - La imagen se carga correctamente desde un archivo DICOM.
        - Se convierte al espacio de color RGB (3 canales).
        - La imagen resultante mantiene el tipo de dato `uint8`.
        - Las dimensiones de salida son correctas `(10, 10, 3)`.
    Args:
        loader.img: contiene el objeto DICOM cargado.
        loader.img_RGB: contiene los datos de imagen convertidos a RGB.
    """
  
    path = "test.dcm"
    mock_img = MagicMock()
    mock_img.pixel_array = np.ones((10, 10), dtype=np.uint8) * 100
    
    with patch("pydicom.dcmread", return_value=mock_img):
        loader = ImageLoader(path)
        img_rgb = loader.get_img_RGB()
        assert img_rgb.shape == (10, 10, 3)
        assert img_rgb.dtype == np.uint8


