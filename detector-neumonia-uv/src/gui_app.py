import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
import csv
from PIL import ImageTk, Image
import tkcap
from integrator import PneumoniaIntegrator


class PneumoniaDetectionApp:
    """
    Clase que gestiona la interfaz gráfica de usuario.
    
    Se encarga únicamente de widgets y eventos, delegando 
    la lógica al integrador.
    """

    def __init__(self):
        """
        Inicializa la ventana principal y los componentes de la interfaz.

        """
        self.root = tk.Tk()
        
        # Integrador reemplaza a predictor
        self.integrator = PneumoniaIntegrator()
        
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("1200x600")
        self.root.resizable(0, 0)

        # Variables de estado solo para GUI
        self.report_id = 0
        self.img1_ref = None
        self.img2_ref = None

        self._setup_ui()
        self.root.mainloop()

    def _setup_ui(self):
        """Configura los elementos visuales de la aplicación."""
        self._set_labels()
        self._set_inputs()
        self._set_buttons()

    def _set_labels(self):
        """Inicializa y posiciona las etiquetas."""
        bold_font = font.Font(weight="bold")
        ttk.Label(self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", 
                  font=bold_font).place(x=122, y=25)
        ttk.Label(self.root, text="Imagen Radiográfica", font=bold_font).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=bold_font).place(x=545, y=65)
        ttk.Label(self.root, text="Cédula Paciente:", font=bold_font).place(x=820, y=300)
        ttk.Label(self.root, text="Resultado:", font=bold_font).place(x=820, y=351)
        ttk.Label(self.root, text="Probabilidad:", font=bold_font).place(x=820, y=400)
    
    def _set_inputs(self):
        """Inicializa y configura los widgets de entrada."""
        self.patient_id = tk.StringVar()
        self.entry_id = ttk.Entry(self.root, textvariable=self.patient_id, width=10)
        self.entry_id.place(x=1000, y=300)

        self.txt_img_orig = tk.Text(self.root, width=31, height=15)
        self.txt_img_orig.place(x=65, y=90)
        
        self.txt_img_heat = tk.Text(self.root, width=31, height=15)
        self.txt_img_heat.place(x=500, y=90)

        self.txt_result = tk.Text(self.root, width=12, height=2)
        self.txt_result.place(x=1000, y=350)
        
        self.txt_proba = tk.Text(self.root, width=12, height=2)
        self.txt_proba.place(x=1000, y=400)

    def _set_buttons(self):
        """Inicializa y configura todos los botones."""
        self.btn_predict = ttk.Button(
            self.root, 
            text="Predecir", 
            state="disabled", 
            command=self.run_prediction
        )
        self.btn_predict.place(x=220, y=460)

        ttk.Button(self.root, text="Cargar Imagen", command=self.load_image).place(x=70, y=460)
        ttk.Button(self.root, text="Guardar", command=self.save_csv).place(x=370, y=460)
        ttk.Button(self.root, text="PDF", command=self.generate_pdf).place(x=520, y=460)
        ttk.Button(self.root, text="Borrar", command=self.clear_fields).place(x=670, y=460)
    
    def load_image(self):
        """Carga imagen usando el integrador."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("DICOM", "*.dcm"), ("Imágenes", "*.jpg *.png *.jpeg"), ("Todos", "*.*"))
        )
        
        if filepath:
            try:
                self.integrator.load_and_prepare_image(filepath)
                img_resized = self.integrator.get_loaded_image().resize((250, 250), Image.LANCZOS)
                self.img1_ref = ImageTk.PhotoImage(img_resized)
                
                self.txt_img_orig.delete("1.0", tk.END)
                self.txt_img_orig.image_create(tk.END, image=self.img1_ref)
                self.btn_predict["state"] = "normal"
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar: {e}")

    def run_prediction(self):
        """Ejecuta predicción usando el integrador."""
        try:
            # Obtener todo del integrador
            result = self.integrator.analyze_image()
            
            label = result['label']
            probability = result['probability']
            heatmap = result['heatmap']
            
            # Mostrar heatmap
            img_heat = Image.fromarray(heatmap).resize((250, 250), Image.LANCZOS)
            self.img2_ref = ImageTk.PhotoImage(img_heat)
            
            self.txt_img_heat.delete("1.0", tk.END)
            self.txt_img_heat.image_create(tk.END, image=self.img2_ref)
            
            # Mostrar textos
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(tk.END, label)
            
            self.txt_proba.delete("1.0", tk.END)
            self.txt_proba.insert(tk.END, f"{probability:.2f}%")
            
        except ValueError as ve:
            messagebox.showwarning("Advertencia", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Predicción falló: {e}")

    def save_csv(self):
        """Guarda los resultados actuales en un archivo CSV."""
        try:
            with open("reports/historial.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="-")
                res = self.txt_result.get("1.0", tk.END).strip()
                prob = self.txt_proba.get("1.0", tk.END).strip()
                w.writerow([self.patient_id.get(), res, prob])
            messagebox.showinfo("Guardar", "Datos guardados con éxito.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {e}")

    def generate_pdf(self):
        """Genera un reporte PDF de la vista actual."""
        try:
            cap = tkcap.CAP(self.root)
            img_path = f"reports/figures/Reporte{self.report_id}.jpg"
            cap.capture(img_path)
            
            pdf_img = Image.open(img_path).convert("RGB")
            pdf_img.save(f"reports/figures/Reporte{self.report_id}.pdf")
            
            self.report_id += 1
            messagebox.showinfo("PDF", "El PDF fue generado con éxito.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el PDF: {e}")

    def clear_fields(self):
        """Limpia la interfaz y el estado."""
        if messagebox.askokcancel("Confirmación", "Se borrarán todos los datos."):
            # Limpiar integrador
            self.integrator.reset()
            
            # Limpiar widgets
            self.entry_id.delete(0, tk.END)
            self.txt_result.delete("1.0", tk.END)
            self.txt_proba.delete("1.0", tk.END)
            self.txt_img_orig.delete("1.0", tk.END)
            self.txt_img_heat.delete("1.0", tk.END)
            
            self.img1_ref = None
            self.img2_ref = None
            
            self.btn_predict["state"] = "disabled"
            messagebox.showinfo("Borrar", "Los datos se borraron con éxito.")
