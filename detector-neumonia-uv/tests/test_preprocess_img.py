import pytest
import numpy as np
from PIL import Image
from src.preprocess_img import ImageProcessor

def test_normalize_for_display_2d():
    # Simulate a single channel image (e.g. DICOM)
    img_array = np.array([[0, 100], [200, 300]], dtype=np.uint16)
    img_bgr, img_pil = ImageProcessor.normalize_for_display(img_array)
    
    assert isinstance(img_pil, Image.Image)
    assert img_bgr.shape == (2, 2, 3)
    assert img_bgr.dtype == np.uint8
    # Max value 300 should be 255 in uint8
    assert img_bgr[1, 1, 0] == 255

def test_normalize_for_display_3d():
    # Simulate a BGR image
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    img_array[0, 0, 0] = 255 # Blue pixel
    img_bgr, img_pil = ImageProcessor.normalize_for_display(img_array)
    
    assert isinstance(img_pil, Image.Image)
    assert img_bgr.shape == (10, 10, 3)
    # Check if conversion to RGB for PIL worked (PIL image should have red at 0,0 if it was BGR)
    # Actually normalize_for_display does:
    # img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    # So if we passed BGR with Blue at 0,0, PIL should have Red at 0,0? No, BGR->RGB
    assert img_pil.getpixel((0, 0)) == (0, 0, 255) # Wait, cv2.cvtColor(BGR, RGB) means Blue(255,0,0) becomes Red(0,0,255)? No.
    # BGR (255, 0, 0) -> RGB (0, 0, 255). Yes.

def test_prepare_for_model():
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ImageProcessor.prepare_for_model(img_array)
    
    # Shape should be (1, 512, 512, 1)
    assert result.shape == (1, 512, 512, 1)
    assert result.dtype == np.float64
    assert np.max(result) <= 1.0

def test_preprocess_delegation():
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    # Preprocess should just call prepare_for_model
    result = ImageProcessor.preprocess(img_array)
    assert result.shape == (1, 512, 512, 1)
