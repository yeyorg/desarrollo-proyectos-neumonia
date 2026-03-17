import os

from integrator import PneumoniaIntegrator


class PneumoniaConsoleApp:
    """Aplicacion de consola para la deteccion de neumonia."""

    def __init__(self):
        self.integrator = PneumoniaIntegrator()

    def run(self):
        """Ejecuta el flujo interactivo en consola."""
        cedula = self._prompt_cedula()
        image_path = self._prompt_image_path()

        self.integrator.load_and_prepare_image(image_path)
        result = self.integrator.analyze_image()

        label = result["label"]
        probability = result["probability"]

        print("\nResultado")
        print(f"Cedula: {cedula}")
        print(f"Prediccion: {label}")
        print(f"Probabilidad: {probability:.2f}%")

    def _prompt_cedula(self):
        """Solicita la cedula hasta que sea valida."""
        while True:
            cedula = input("Ingrese la cedula del paciente: ").strip()
            if cedula:
                return cedula
            print("La cedula no puede estar vacia. Intente de nuevo.")

    def _prompt_image_path(self):
        """Solicita la ruta de la imagen hasta que exista."""
        while True:
            image_path = input("Ingrese la ruta de la imagen: ").strip().strip('"')
            if os.path.isfile(image_path):
                return image_path
            print("La ruta no existe o no es un archivo. Intente de nuevo.")
