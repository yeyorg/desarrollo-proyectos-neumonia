from preprocess_img import ImagePreprocessor
from grad_cam import GradCam
import numpy as np

class Predictor:
    """
    Predict pneumonia type from an X-ray image.
    Args:
        array: Input X-ray image as a numpy array or compatible format.
    Returns:
        tuple: A tuple containing:
            - label (str): Predicted pneumonia type ("bacteriana", "normal", or "viral")
            - proba (float): Confidence score as a percentage (0-100)
            - heatmap: Grad-CAM visualization highlighting model attention regions
    """
    def __init__(self, model):
        self.model = model
        self.label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    
    def predict(self, array):
        batch_array_img = ImagePreprocessor.preprocess(array)
        
        prediction_array = self.model.predict(batch_array_img)
        prediction = np.argmax(prediction_array)
        proba = np.max(prediction_array) * 100
        
        label = self.label_map.get(prediction, "desconocida")
        
        heatmap = grad_cam(array, prediction)
        
        return (label, proba, heatmap)