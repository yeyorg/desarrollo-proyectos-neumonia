import argparse

from console_app import PneumoniaConsoleApp
from gui_app import PneumoniaDetectionApp

def main():
    """Lanza la aplicacion en modo GUI o consola."""

    parser = argparse.ArgumentParser(description="Detector de neumonia")
    parser.add_argument(
        "--console",
        action="store_true",
        help="Ejecuta la aplicacion por consola",
    )
    args = parser.parse_args()

    if args.console:
        PneumoniaConsoleApp().run()
    else:
        PneumoniaDetectionApp()


if __name__ == "__main__":
    main()
