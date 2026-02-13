
import tensorflow as tf


class ModelLoader:
    """Carga y mantiene en memoria el modelo entrenado."""

    def __init__(self, model_path="models/conv_MLP_84.h5"):
        """
        Inicializa el cargador de modelos y carga el modelo.
        
        Args:
            model_path (str): Ruta al archivo del modelo en formato .h5
        """
        self.path = model_path
        self.model = self._load()

    def _load(self):
        """Método privado para la carga segura del modelo."""
        return tf.keras.models.load_model(self.path, compile=False)

    def get_model(self):
        """Retorna la instancia del modelo cargado.
        
        Returns:
            tf.keras.Model: El modelo cargado listo para usar.
        """
        return self.model