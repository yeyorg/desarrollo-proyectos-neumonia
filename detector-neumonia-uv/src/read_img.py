import cv2
import pydicom as dicom
import numpy as np
from PIL import Image


class ImageLoader:
    """
    Clase encargada de la carga de imágenes desde el sistema de archivos.
    Soporta formatos DICOM y genera representaciones PIL y RGB para visualización.
    
    Attributes:
        img: Objeto pydicom Dataset con los datos DICOM cargados
        img2show: Imagen PIL para visualización directa
        img_RGB: Array numpy BGR normalizado para procesamiento con OpenCV
    """

    def __init__(self, path):
        """
        Inicializa el cargador de imágenes y procesa un archivo DICOM.
        
        Lee el archivo DICOM especificado y genera automáticamente dos 
        representaciones de la imagen: una para visualización (PIL) y 
        otra normalizada en formato RGB.
        
        Args:
            path (str): Ruta al archivo DICOM a cargar
        """
        self.img = dicom.dcmread(path)
        self._generate_img_to_show()
        self._generate_img_RGB()
    
    def _generate_img_to_show(self):
        """
        Genera una imagen PIL a partir del pixel_array DICOM.
        
        Convierte el array de píxeles del DICOM directamente a un objeto 
        PIL Image sin normalización, preservando los valores originales 
        para visualización.
        """
        img_array = self.img.pixel_array
        self.img2show = Image.fromarray(img_array)
    
    def _generate_img_RGB(self):
        """
        Genera una imagen RGB normalizada del DICOM.
        
        Realiza normalización min-max escalando los valores de píxeles 
        al rango 0-255, convierte a uint8 y transforma de escala de grises 
        a RGB de 3 canales para compatibilidad con OpenCV.
        """
        img_array = self.img.pixel_array
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        self.img_RGB = img_RGB
    
    def get_img_RGB(self):
        """
        Obtiene la imagen RGB normalizada.
        
        Returns:
            numpy.ndarray: Array BGR (512x512x3) normalizado a rango 0-255, 
                          listo para procesamiento con OpenCV
        """
        return self.img_RGB
    
    def get_img_to_show(self):
        """
        Obtiene la imagen PIL para visualización.
        
        Returns:
            PIL.Image: Objeto Image con los valores de píxeles originales 
                      del DICOM, adecuado para mostrar con bibliotecas 
                      como matplotlib o Tkinter
        """
        return self.img2show
