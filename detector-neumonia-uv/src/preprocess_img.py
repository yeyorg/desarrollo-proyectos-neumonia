"""
Preprocesamiento de imágenes
"""

import cv2
import numpy as np

class ImagePreprocessor:
    """Clase para preprocesar imágenes de rayos X antes de predicción."""
    
    @staticmethod
    def preprocess(array):
        """
        Preprocesa imagen: resize, CLAHE, normalización.
        
        Args:
            array: numpy array BGR de la imagen
            
        Returns:
            numpy array preprocesado (1, 512, 512, 1)
        """
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        
        array = array / 255.0
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array

