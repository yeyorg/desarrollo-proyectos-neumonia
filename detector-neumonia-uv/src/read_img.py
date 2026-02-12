import cv2
import pydicom as dicom
from PIL import Image


class ImageLoader:
    """
    Clase encargada ÚNICAMENTE de la carga de imágenes desde el sistema de archivos.
    Soporta formatos DICOM y JPG/PNG/JPEG.
    """

    @staticmethod
    def read_dicom_file(path):
        """
        Lee un archivo DICOM y devuelve el pixel_array original.
        """
        img = dicom.dcmread(path)
        return img.pixel_array

    @staticmethod
    def read_jpg_file(path):
        """
        Lee un archivo de imagen estándar (JPG/PNG) usando OpenCV.
        Devuelve la imagen en formato BGR (default de OpenCV).
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen en la ruta: {path}")
        return img

    @classmethod
    def load(cls, path):
        """
        Método principal para cargar una imagen basado en su extensión.
        """
        if path.lower().endswith(('.dcm', '.dicom')):
            return cls.read_dicom_file(path)
        else:
            return cls.read_jpg_file(path)