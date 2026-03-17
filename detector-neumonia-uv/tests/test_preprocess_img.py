import pytest
import numpy as np
from src.preprocess_img import ImagePreprocessor

def test_preprocess_shape():
    """
    Test que verifica que la función preprocess retorna las dimensiones correctas.
    Verifica que:
    - La imagen de entrada (480, 640, 3) en formato RGB
    - Se redimensiona correctamente a (512, 512)
    - Se convierte a escala de grises (canal único)
    - Se añade dimensión de batch
    - El resultado final tiene forma (1, 512, 512, 1)
    """
    img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = ImagePreprocessor.preprocess(img_array)
    
    # Shape should be (1, 512, 512, 1)
    assert result.shape == (1, 512, 512, 1)

def test_preprocess_normalization():
    """
    Test que verifica la normalización correcta de imágenes en el preprocesamiento.
    Valida que después de aplicar el preprocesamiento a una imagen, los valores
    de píxeles se encuentren en el rango normalizado [0.0, 1.0].
    Casos verificados:
        - Valor máximo de la imagen normalizada es <= 1.0
        - Valor mínimo de la imagen normalizada es >= 0.0
    Nota:
        Utiliza una imagen de prueba con todos los píxeles en valor máximo (255)
        para verificar que la normalización se aplica correctamente.
    """
    img_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = ImagePreprocessor.preprocess(img_array)
    
    # Valores deben estar en rango 0-1 después de normalización
    assert np.max(result) <= 1.0
    assert np.min(result) >= 0.0

def test_preprocess_dtype():
    """
    Test que verifica que el método preprocess retorna una imagen normalizada en formato float.
    Este test valida que la entrada de una imagen en formato uint8 sea correctamente
    convertida a un tipo de dato flotante (float32 o float64) durante el preprocesamiento,
    lo que es esencial para el posterior procesamiento del modelo de detección de neumonía.
    Verifica:
        - Que el dtype del resultado sea float32 o float64
        - Que la conversión de tipo se ejecute correctamente en arrays de forma (100, 100, 3)
    """
    
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ImagePreprocessor.preprocess(img_array)
    
    # dtype should be float
    assert result.dtype in [np.float32, np.float64]

def test_preprocess_resizes():
    """
    Verifica que el preprocesamiento de imágenes redimensiona correctamente 
    a un tamaño estándar de 512x512 píxeles, independientemente del tamaño 
    de entrada.
    Casos de prueba:
    - Imagen pequeña (64x64): valida que se amplíe a 512x512
    - Imagen grande (1024x1024): valida que se reduzca a 512x512
    Se comprueba que la salida tiene la forma esperada: 
    (batch_size=1, height=512, width=512, channels=1)
    """
    
    # Imagen pequeña
    img_small = np.ones((64, 64, 3), dtype=np.uint8) * 100
    result_small = ImagePreprocessor.preprocess(img_small)
    assert result_small.shape == (1, 512, 512, 1)
    
    # Imagen grande
    img_large = np.ones((1024, 1024, 3), dtype=np.uint8) * 100
    result_large = ImagePreprocessor.preprocess(img_large)
    assert result_large.shape == (1, 512, 512, 1)

