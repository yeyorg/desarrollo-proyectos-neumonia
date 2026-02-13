"""
Módulo integrador que coordina la carga, preprocesamiento y predicción.
"""

from read_img import ImageLoader
from predictor import Predictor


class PneumoniaIntegrator:
    """
    Coordinador que unifica carga de imagen y predicción.
    Retorna label, probabilidad y heatmap de forma unificada.
    """
    
    def __init__(self):
        """
        Inicializa el integrador cargando el modelo y el predictor.
        """
        self.predictor = Predictor()
        self.current_array = None
    
    def load_and_prepare_image(self, filepath):
        """
        Carga imagen desde archivo y la prepara para visualización.
        
        Args:
            filepath: Ruta del archivo (DICOM, JPG, PNG).
            
        Returns:
            tuple: (img_array_RGB, img_PIL_for_display)
        """
        loader = ImageLoader(filepath)
        self.current_array = loader.get_img_RGB()
        self.img_to_show = loader.get_img_to_show()
        
    def analyze_image(self):
        """
        Ejecuta predicción y genera heatmap.
        
        Args:
            img_array: numpy array (opcional, usa el último cargado si es None).
            
        Returns:
            dict: {
                'label': str,           # 'bacteriana', 'normal', 'viral'
                'probability': float,   # ej: 94.25
                'heatmap': ndarray      # RGB (512, 512, 3)
            }
        """
        if self.current_array is None:
            raise ValueError("No hay imagen cargada.")
        
        label, probability, heatmap = self.predictor.predict(self.current_array)
        
        return {
            'label': label,
            'probability': probability,
            'heatmap': heatmap
        }
    
    def reset(self):
        """Limpia el array almacenado."""
        self.current_array = None

    def get_image(self):
        """Retorna la imagen preparada para mostrar."""
        return self.img_to_show
