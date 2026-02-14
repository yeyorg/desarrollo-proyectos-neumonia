import numpy as np
import cv2
import tensorflow as tf

class GradCAMGenerator:
    """
    Generador de mapas de calor usando técnica Grad-CAM.
    
    Implementa Class Activation Mapping con gradientes para visualizar
    las regiones más importantes de una imagen según el modelo de red neuronal.
    El preprocesamiento es delegado a una función externa para mayor flexibilidad.
    
    Attributes:
        model: Modelo de TensorFlow/Keras cargado.
    """
    
    def __init__(self, model, target_layer_name="conv10_thisone", output_size=(512, 512)):
        """
        Inicializa el generador de Grad-CAM.
        
        Args:
            model: Modelo de TensorFlow/Keras entrenado.
            target_layer_name (str): Nombre de la capa convolucional objetivo para Grad-CAM.
                Por defecto "conv10_thisone".
            output_size (tuple): Tupla (ancho, alto) para el tamaño de salida del heatmap.
                Por defecto (512, 512).
                
        Raises:
            ValueError: Si el modelo es None, la capa no existe, o los parámetros son inválidos.
        """
        if model is None:
            raise ValueError("No se recibió un modelo válido")
        
        self.model = model
        self.target_layer_name = target_layer_name
        self.output_size = output_size
        self.expected_channels = 1  # Canal esperado para imágenes de entrada
        
        # Validar que la capa objetivo existe en el modelo
        self._validate_target_layer()
    
    
    def _compute_gradients(self, preprocessed_img, predicted_class):
        """
        Calcula los gradientes necesarios para Grad-CAM.
        
        Construye un modelo que extrae las salidas de la última capa convolucional
        y la salida final, luego calcula los gradientes respecto a la clase predicha.
        
        Args:
            preprocessed_img (np.ndarray): Imagen preprocesada con shape (1, 512, 512, 1).
            predicted_class (int): Índice de la clase para la cual calcular gradientes.
            
        Returns:
            tuple: (conv_outputs, grads) donde:
                - conv_outputs (np.ndarray): Salidas de la última capa convolucional.
                - grads (np.ndarray): Gradientes respecto a la clase.
        """
        # Crear modelo de gradientes
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer("conv10_thisone").output,
                self.model.output
            ]
        )
        
        # Calcular gradientes usando GradientTape
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_img)
            
            # Manejo de predicciones en formato lista
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            # Calcular pérdida para la clase predicha
            loss = predictions[:, predicted_class]
        
        # Calcular gradientes
        grads = tape.gradient(loss, conv_outputs)
        
        return conv_outputs, grads
    
    def _generate_heatmap_matrix(self, conv_outputs, grads):
        """
        Genera la matriz numérica del heatmap usando Grad-CAM.
        
        Calcula los gradientes ponderados por canales para obtener una matriz
        que represente la importancia de cada región espacial.
        
        Args:
            conv_outputs (tf.Tensor): Salidas de la última capa convolucional.
            grads (tf.Tensor): Gradientes calculados respecto a la clase.
            
        Returns:
            np.ndarray: Matriz 2D normalizada en rango [0, 1] de shape (H, W).
        """
        # Promediar gradientes a través de dimensiones espaciales
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        
        # Convertir outputs a numpy
        conv_outputs_np = conv_outputs[0].numpy()
        
        # Multiplicar cada canal por su peso (importancia del gradiente)
        weighted_outputs = conv_outputs_np.copy()
        for i in range(pooled_grads.shape[0]):
            weighted_outputs[:, :, i] *= pooled_grads[i]
        
        # Promediar canales ponderados
        heatmap = np.mean(weighted_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU: descartar valores negativos
        
        # Normalizar a rango [0, 1]
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        return heatmap
    
    def _colorize_heatmap(self, heatmap_matrix):
        """
        Coloriza una matriz de heatmap en formato BGR.
        
        Redimensiona la matriz al tamaño objetivo, la convierte a 8-bit,
        aplica un colormap y retorna la imagen coloreada.
        
        Args:
            heatmap_matrix (np.ndarray): Matriz 2D del heatmap normalizada en [0, 1].
            
        Returns:
            np.ndarray: Imagen coloreada en formato BGR con shape (H, W, 3).
        """
        # Redimensionar a tamaño de salida
        heatmap_resized = cv2.resize(heatmap_matrix, 
                                      (512, 512))
        
        # Convertir a 8-bit [0, 255]
        heatmap_8bit = np.uint8(255 * heatmap_resized)
        
        # Aplicar colormap JET para visualización
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def _compose_visualization(self, colored_heatmap, original_array):
        """
        Superpone el heatmap coloreado sobre la imagen original.
        
        Redimensiona la imagen original, aplica transparencia al heatmap
        y combina ambas imágenes, retornando el resultado en formato RGB.
        
        Args:
            colored_heatmap (np.ndarray): Imagen heatmap coloreada en BGR.
            original_array (np.ndarray): Imagen original en BGR.
            
        Returns:
            np.ndarray: Imagen compuesta en formato RGB.
        """
        # Redimensionar imagen original
        original_resized = cv2.resize(original_array, 
                                      (512, 512))
        
        # Aplicar factor de transparencia al heatmap
        heatmap_transparent = (colored_heatmap * 0.8).astype(np.uint8)
        
        # Superponer: añadir heatmap transparente sobre imagen original
        superimposed = cv2.add(heatmap_transparent, original_resized).astype(np.uint8)
        
        # Convertir de BGR a RGB
        superimposed_rgb = superimposed[:, :, ::-1]
        
        return superimposed_rgb
    
    def generate(self, array, predicted_class, preprocessed_img):
        """
        Genera el mapa de calor Grad-CAM para una imagen y clase predicha.
        
        Orquesta todo el proceso: preprocesamiento, cálculo de gradientes,
        y generación del heatmap superpuesto.
        
        Args:
            array (np.ndarray): Imagen de entrada en formato BGR.
            predicted_class (int): Índice de la clase para visualizar.
            
        Returns:
            np.ndarray: Imagen con heatmap superpuesto en formato RGB con shape (512, 512, 3).
            
        Raises:
            ValueError: Si predicted_class no es un entero válido.
        """
        # Validar entrada
        try:
            predicted_class = int(predicted_class)
        except (ValueError, TypeError):
            raise ValueError(f"la clase predicha debe ser un entero, se recibió: {type(predicted_class)}")
        if preprocessed_img.shape != (1, 512, 512, 1):
            raise ValueError(f"preprocessed_img debe tener shape (1, 512, 512, 1), se recibió: {preprocessed_img.shape}")
        
        # Calcular gradientes
        conv_outputs, grads = self._compute_gradients(preprocessed_img, predicted_class)
        
        # Generar visualización Grad-CAM en pasos secuenciales
        heatmap_matrix = self._generate_heatmap_matrix(conv_outputs, grads)
        colored_heatmap = self._colorize_heatmap(heatmap_matrix)
        visualization = self._compose_visualization(colored_heatmap, array)
        
        return visualization