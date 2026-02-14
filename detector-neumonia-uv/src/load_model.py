import tensorflow as tf
import os


class ModelLoader:
    """Carga y mantiene en memoria el modelo entrenado."""

    def __init__(self, model_path="models/conv_MLP_84.h5"):
        """
        Inicializa el cargador de modelos y carga el modelo.
        
        Args:
            model_path (str): Ruta al archivo del modelo en formato .h5
            
        Raises:
            FileNotFoundError: Si el archivo del modelo no existe
            ValueError: Si el modelo no se puede cargar o es inválido
        """
        self.path = model_path
        self._validate_file_exists()
        self.model = self._load()
        self._validate_model_integrity()

    def _validate_file_exists(self):
        """Valida que el archivo del modelo exista."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Modelo no encontrado en: {self.path}")

    def _load(self):
        """Método privado para la carga segura del modelo."""
        try:
            return tf.keras.models.load_model(self.path, compile=False)
        except Exception as e:
            raise ValueError(f"Error al cargar el modelo: {str(e)}")

    def _validate_model_integrity(self):
        """
        Valida la integridad del modelo después de cargarlo.
        
        Verifica que:
        - El modelo tenga capas
        - La arquitectura sea válida
        - Se pueda hacer una predicción de prueba
        
        Raises:
            ValueError: Si el modelo no pasa las validaciones
        """
        # Verificar que tenga capas
        if not self.model.layers:
            raise ValueError("El modelo no contiene capas")
        
        # Verificar input shape
        if not hasattr(self.model, 'input_shape'):
            raise ValueError("El modelo no tiene input_shape definido")
        
        # Prueba de predicción con datos dummy
        try:
            input_shape = self.model.input_shape
            # Crear array de prueba con la forma correcta
            dummy_input = tf.random.normal(
                shape=(1,) + input_shape[1:]
            )
            _ = self.model.predict(dummy_input, verbose=0)
        except Exception as e:
            raise ValueError(f"El modelo falla al hacer predicción: {str(e)}")

    def get_model(self):
        """
        Retorna la instancia del modelo cargado.
        
        Returns:
            tf.keras.Model: El modelo cargado y validado listo para usar.
        """
        return self.model
