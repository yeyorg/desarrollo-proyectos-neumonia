#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING

import csv

from PIL import ImageTk, Image

import tkcap
import numpy as np
import cv2
import pydicom as dicom
import tensorflow as tf

from load_model import ModelLoader
from gui_app import PneumoniaDetectionApp
model = ModelLoader().get_model()

def grad_cam(array, predicted_class):
    img = preprocess(array)
    
    
    # Convertir a entero de Python
    predicted_class = int(predicted_class)
    
    # Obtener la última capa convolucional
    last_conv_layer_name = "conv10_thisone"
    
    # Crear modelo de gradientes
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Calcular gradientes usando GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        if isinstance(predictions, list):
            predictions = predictions[0]
        loss = predictions[:, predicted_class]
    
    # Calcular gradientes de la salida con respecto a la última capa conv
    grads = tape.gradient(loss, conv_outputs)
    
    # Promediar gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada canal por su importancia
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Crear el heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer heatmap en imagen original
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    
    return superimposed_img[:, :, ::-1]


def predict(array):
    # 1. Preprocesar imagen
    batch_array_img = preprocess(array)
    
    # 2. Cargar modelo y predecir UNA SOLA VEZ
    prediction_array = model.predict(batch_array_img)
    prediction = np.argmax(prediction_array)
    proba = np.max(prediction_array) * 100
    
    # 3. Obtener etiqueta
    label = ""
    if prediction == 0:
        label = "bacteriana"
    elif prediction == 1:
        label = "normal"
    elif prediction == 2:
        label = "viral"
    
    # 4. Generar Grad-CAM pasando la clase predicha
    heatmap = grad_cam(array, prediction)
    
    return (label, proba, heatmap)


def read_dicom_file(path):
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array



def main():
    """Inicializa la raíz de Tkinter y lanza la aplicación."""

    # 1. Crear la instancia de la ventana raíz
    root = tk.Tk()
    
    # 2. Instanciar la aplicación (La clase se encarga de configurar la GUI)
    PneumoniaDetectionApp(root)
    
    # 3. Iniciar el bucle de eventos
    root.mainloop()


if __name__ == "__main__":
    main()
