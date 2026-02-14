# Sistema de Apoyo al Diagnóstico Médico de Neumonía

Este proyecto utiliza Deep Learning para el procesamiento de imágenes radiográficas de tórax en formato **DICOM** con el fin de clasificarlas y proporcionar herramientas de explicabilidad visual para el personal médico.

---

## 📌 Evolución del Proyecto

### Versión 1: Prueba de Concepto Original
Desarrollado inicialmente por **Isabella Torres Revelo** y **Nicolas Diaz Salazar** ([Repositorio Original](https://github.com/yeyorg/UAO-Neumonia)). Esta versión sentó las bases científicas utilizando modelos de Deep Learning y técnicas de Grad-CAM para la detección de:
1. Neumonía Bacteriana
2. Neumonía Viral
3. Sin Neumonía (Normal)

### Versión 2: Refactorización Profesional (Actual)
La versión actual representa una evolución orientada a la ingeniería de software aplicada, centrada en la robustez y escalabilidad del sistema. 

#### Mejoras Arquitectónicas Detalladas:

*   **Desacoplamiento (Low Coupling):** Se eliminó la dependencia directa entre la interfaz gráfica (GUI) y la lógica de Deep Learning. Mientras que en la V1 la interfaz gestionaba procesos de ejecución del modelo, en la **V2** se implementó una **Capa de Integración** (`PneumoniaIntegrator`). Este patrón actúa como un mediador, permitiendo que la interfaz (`gui_app.py`) solo se preocupe por la visualización, mientras que la lógica de inferencia reside en módulos independientes. Beneficio: Facilidad para actualizar el modelo de IA o cambiar el motor gráfico sin romper el sistema completo.
*   **Alta Cohesión (High Cohesion):** Se aplicó el **Principio de Responsabilidad Única (SRP)**, redistribuyendo el código en componentes especializados:
    *   `ImageLoader`: Se encarga exclusivamente de la lectura y validación de archivos **DICOM**.
    *   `ImagePreprocessor`: Centraliza las transformaciones matemáticas, normalización y ecualización (CLAHE), asegurando que el modelo reciba datos consistentes.
    *   `Predictor`: Aísla la complejidad de la inferencia, gestionando la carga del modelo y la interpretación de los tensores de salida.
    *   `GradCAMGenerator`: Encapsula la lógica de generación de mapas de calor para explicabilidad.
*   **Mantenibilidad y Gestión de Dependencias:** Se migró el sistema al gestor de paquetes moderno **`uv`**, garantizando entornos reproducibles y una instalación de dependencias mucho más rápida y segura.

---

## ✨ Funcionalidades y Beneficios

| Funcionalidad | Beneficio para el Usuario |
| :--- | :--- |
| **Soporte DICOM** | Permite trabajar directamente con formatos estándar hospitalarios. |
| **Predicción Automatizada** | Acelera el triaje médico mediante un diagnóstico preliminar basado en redes convolucionales. |
| **Mapas de Calor (Grad-CAM)** | Aporta transparencia al "caja negra" de la IA, permitiendo al médico validar visualmente las zonas pulmonares afectadas. |
| **Generación de Reportes PDF** | Facilita la documentación y el intercambio de resultados entre especialistas de forma profesional. |
| **Base de Datos Histórica (CSV)** | Permite llevar un registro organizado de los pacientes procesados para seguimiento. |

---

## 🚀 Instalación Local (V2)

El proyecto ahora utiliza **`uv`** para una gestión de dependencias eficiente.

1.  **Requisitos:**
    *   Python 3.12+
    *   [Instalar uv](https://github.com/astral-sh/uv)

2.  **Preparación y Ejecución:**
    ```bash
    # Clonar el proyecto
    git clone https://github.com/yeyorg/desarrollo-proyectos-neumonia.git
    cd desarrollo-proyectos-neumonia/detector-neumonia-uv

    # Instalar dependencias y ejecutar en un solo paso
    uv run python src/main.py
    ```

---

## 📂 Estructura de Módulos (V2)

```text
src/
├── main.py            # Punto de entrada de la aplicación
├── gui_app.py         # Interfaz gráfica (Tkinter) - Solo lógica visual
├── integrator.py      # Coordinador entre GUI y lógica de predicción
├── predictor.py       # Orquestador de inferencia y Grad-CAM
├── read_img.py        # Módulo de carga (ImageLoader)
├── preprocess_img.py  # Módulo de pre-procesamiento (ImagePreprocessor)
├── load_model.py      # Gestor de carga del modelo conv_MLP_84.h5
└── grad_cam.py        # Generador de explicabilidad visual
```

---

## 🧠 Detalles Técnicos

### El Modelo: conv_MLP_84
Basado en arquitecturas eficientes para rayos X de tórax, el modelo consta de **5 bloques convolucionales** con conexiones residuales (*skip connections*) que evitan el desvanecimiento del gradiente. Utiliza **16 a 80 filtros** progresivos y capas densas finales de alta capacidad (1024 neuronas) para una clasificación precisa.

### Grad-CAM (Gradient-weighted Class Activation Mapping)
Técnica que calcula el gradiente de la salida de la clase predicha con respecto a la última capa convolucional. Esto genera un mapa de calor que resalta las regiones de la radiografía que más influyeron en la decisión de la red neuronal, permitiendo una validación clínica cualitativa.

---

## 👥 Créditos

*   **V1 (Original):** Isabella Torres Revelo & Nicolas Diaz Salazar.
*   **V2 (Refactorización):** Equipo de Desarrollo Especialización IA - UAO.

---

## ⚖️ Licencia y Uso Educativo

Este proyecto se distribuye bajo la licencia **Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)**. 

**¿Qué significa esto?**
*   **Usted es libre de:** Compartir, copiar y redistribuir el material en cualquier medio o formato. Adaptar, remezclar, transformar y construir sobre el material.
*   **Bajo los siguientes términos:**
    *   **Atribución:** Debe dar crédito de manera adecuada y proporcionar un enlace a la licencia e indicar si se han realizado cambios.
    *   **No Comercial:** No puede utilizar el material con fines comerciales o de lucro.

---

## ⚠️ Descargo de Responsabilidad Médico (Medical Disclaimer)

**IMPORTANTE:** Este software es una herramienta de carácter **estrictamente educativo y experimental**. 

1.  **No para diagnóstico clínico:** Los resultados y predicciones generados por este sistema **NO** deben ser utilizados como base para diagnósticos médicos reales, tratamientos o decisiones clínicas de ningún tipo.
2.  **Validación Profesional:** Cualquier evaluación radiográfica debe ser realizada y validada por un médico radiólogo o profesional de la salud calificado.
3.  **Sin Garantía:** El software se proporciona "tal cual", sin garantías de precisión, confiabilidad o idoneidad para un propósito médico específico. Los desarrolladores no se hacen responsables de ninguna decisión tomada basándose en el uso de esta herramienta.
