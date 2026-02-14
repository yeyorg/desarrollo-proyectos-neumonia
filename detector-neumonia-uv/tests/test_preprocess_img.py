import pytest
import numpy as np
from src.preprocess_img import ImagePreprocessor

def test_preprocess_shape():
    """Test que preprocess retorna la forma correcta."""
    img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = ImagePreprocessor.preprocess(img_array)
    
    # Shape should be (1, 512, 512, 1)
    assert result.shape == (1, 512, 512, 1)

def test_preprocess_normalization():
    """Test que preprocess normaliza correctamente."""
    img_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = ImagePreprocessor.preprocess(img_array)
    
    # Valores deben estar en rango 0-1 después de normalización
    assert np.max(result) <= 1.0
    assert np.min(result) >= 0.0

def test_preprocess_dtype():
    """Test que preprocess retorna float."""
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ImagePreprocessor.preprocess(img_array)
    
    # dtype should be float
    assert result.dtype in [np.float32, np.float64]

def test_preprocess_resizes():
    """Test que preprocess redimensiona a 512x512."""
    # Imagen pequeña
    img_small = np.ones((64, 64, 3), dtype=np.uint8) * 100
    result_small = ImagePreprocessor.preprocess(img_small)
    assert result_small.shape == (1, 512, 512, 1)
    
    # Imagen grande
    img_large = np.ones((1024, 1024, 3), dtype=np.uint8) * 100
    result_large = ImagePreprocessor.preprocess(img_large)
    assert result_large.shape == (1, 512, 512, 1)

