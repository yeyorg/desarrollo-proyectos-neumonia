import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    """
    Clase encargada de las transformaciones y procesamiento de imágenes.
    Libera a ImageLoader de la responsabilidad de normalización y conversión.
    """

    @staticmethod
    def normalize_for_display(img_array):
        """
        Normaliza un array (DICOM o BGR) para su visualización en una interfaz.
        Devuelve:
            - PIL Image (RGB) para mostrar en la UI.
            - Array (BGR uint8) para procesamiento posterior en OpenCV.
        """
        # Si la imagen es DICOM (posiblemente float o > 8 bit)
        img_float = img_array.astype(float)
        img_max = img_float.max()
        
        if img_max > 0:
            img_normalized = (np.maximum(img_float, 0) / img_max) * 255.0
        else:
            img_normalized = np.zeros_like(img_float)
            
        img_uint8 = np.uint8(img_normalized)
        
        # Caso DICOM (1 canal o ya procesado por ImageLoader)
        if len(img_uint8.shape) == 2:
            # Para PIL
            img_pil = Image.fromarray(img_uint8)
            # Para OpenCV
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        else:
            # Caso JPG/PNG (ya es BGR)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_bgr = img_uint8

        return img_bgr, img_pil

    @staticmethod
    def prepare_for_model(array):
        """
        Realiza el preprocesamiento final (CLAHE, resize) para la red neuronal.
        """
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array


    @staticmethod
    def preprocess(array):
        """
        Método de conveniencia que delega a prepare_for_model.
        """
        return ImageProcessor.prepare_for_model(array)