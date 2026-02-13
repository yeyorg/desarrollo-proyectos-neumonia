import numpy as np

from preprocess_img import ImagePreprocessor
from grad_cam import GradCAMGenerator


class Predictor:
    """Predictor de tipos de neumonía desde imágenes de rayos X.

    Utiliza un modelo entrenado de red neuronal convolucional para
    realizar predicciones de tipos de neumonía (bacteriana, normal, viral)
    y genera visualizaciones Grad-CAM.

    Attributes:
        model (tf.keras.Model): Modelo entrenado para predicción.
        grad_cam (GradCAMGenerator): Instancia de GradCAMGenerator para visualización.
        label_map (dict): Mapeo de índices a etiquetas de neumonía.
    """

    def __init__(self, model):
        """Inicializa el predictor con un modelo entrenado.

        Args:
            model: Modelo de Keras entrenado para predicción de neumonía.

        Raises:
            ValueError: Si el modelo es None o no es válido.
        """
        if model is None:
            raise ValueError("El modelo no puede ser None.")

        self.model = model
        self.grad_cam = GradCAMGenerator(model)
        self.label_map = {
            0: "bacteriana",
            1: "normal",
            2: "viral"
        }

    def predict(self, image_array: np.ndarray):
        """Realiza una predicción de neumonía para una imagen.

        Args:
            image_array: Array numpy con la imagen de rayos X
                en formato (altura, ancho, canales).

        Returns:
            Tupla con (etiqueta, confianza, heatmap):
                - etiqueta (str): Tipo predicho ("bacteriana", "normal", "viral").
                - confianza (float): Puntuación de confianza (0-100).
                - heatmap (np.ndarray): Visualización Grad-CAM de las áreas
                  de influencia en la predicción.

        Raises:
            ValueError: Si image_array es None o está vacío.
        """
        if image_array is None or image_array.size == 0:
            raise ValueError("image_array no puede ser None o estar vacío.")

        # Preprocesar imagen
        batch_array_img = ImagePreprocessor.preprocess(image_array)

        # Realizar predicción
        prediction_array = self.model.predict(batch_array_img, verbose=0)
        prediction_idx = np.argmax(prediction_array)
        confidence = float(np.max(prediction_array) * 100)

        # Obtener etiqueta
        label = self.label_map.get(prediction_idx, "desconocida")

        # Generar visualización Grad-CAM
        heatmap = self.grad_cam.generate(image_array, prediction_idx, batch_array_img)

        return (label, confidence, heatmap)