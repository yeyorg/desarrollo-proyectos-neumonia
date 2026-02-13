from load_model import ModelLoader
from gui_app import PneumoniaDetectionApp

model = ModelLoader().get_model()

def main():
    """Inicializa la raíz de Tkinter y lanza la aplicación."""

    PneumoniaDetectionApp(model)


if __name__ == "__main__":
    main()
