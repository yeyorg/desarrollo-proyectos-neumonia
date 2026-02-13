import cv2
import pydicom as dicom
import numpy as np
from PIL import Image


class ImageLoader:
    """
    Clase encargada de la carga de imágenes desde el sistema de archivos.
    Soporta formatos DICOM.
    """

    def __init__(self,path):
        """
        Lee un archivo DICOM.
        """
        self.img = dicom.dcmread(path)
        self._generate_img_to_show()
        self._generate_img_RGB()
    
    def _generate_img_to_show(self):
        img_array = self.img.pixel_array
        self.img2show = Image.fromarray(img_array)
    
    def _generate_img_RGB(self):
        img_array = self.img.pixel_array
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        self.img_RGB = img_RGB
    
    def get_img_RGB(self):
        return self.img_RGB
    
    def get_img_to_show(self):
        return self.img2show