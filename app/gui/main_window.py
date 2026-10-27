import customtkinter as ctk
from tkinter import filedialog, messagebox

from app.core import image_ops
from app.gui.image_panel import PanelImagen
from app.gui.info_panel import PanelInformacion
from app.gui.sidebar import BarraLateral
from app.tps.tp0_basic import ControladorTP0, SECCION_MENU
from app.tps.tp1_operadores import ControladorTP1
from app.tps.tp2_filtros import ControladorTP2
from app.tps.tp3_canny import ControladorTP3

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VentanaPrincipal(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PIVC - Procesamiento de Imágenes y Visión por Computadora")
        self.geometry("1200x740")
        self.minsize(1000, 640)

        self.imagen_original = None
        self.imagen_resultado = None
        self.ultima_seleccion = None

        self.escuchas_click_pixel = []
        self.escuchas_seleccion_region = []

        self._construir_diseno()
        self._registrar_tps()

    def _construir_diseno(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.barra_lateral = BarraLateral(self, width=260, label_text="")
        self.barra_lateral.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 4), pady=0)

        area_imagenes = ctk.CTkFrame(self, fg_color="transparent")
        area_imagenes.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        area_imagenes.grid_columnconfigure((0, 1), weight=1)
        area_imagenes.grid_rowconfigure(0, weight=1)

        self.panel_original = PanelImagen(
            area_imagenes, titulo="Imagen Original",
            al_hacer_click_pixel=lambda x, y: self._manejar_click_pixel("original", x, y),
            al_seleccionar_region=lambda x1, y1, x2, y2: self._manejar_seleccion_region("original", x1, y1, x2, y2),
        )
        self.panel_original.grid(row=0, column=0, sticky="n", padx=6, pady=6)

        self.panel_resultado = PanelImagen(
            area_imagenes, titulo="Imagen Resultado",
            al_hacer_click_pixel=lambda x, y: self._manejar_click_pixel("resultado", x, y),
            al_seleccionar_region=lambda x1, y1, x2, y2: self._manejar_seleccion_region("resultado", x1, y1, x2, y2),
        )
        self.panel_resultado.grid(row=0, column=1, sticky="n", padx=6, pady=6)

        self.panel_informacion = PanelInformacion(self)
        self.panel_informacion.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))

    def _registrar_tps(self):
        self.barra_lateral.grupo_actual = "TP0"
        self.tp0 = ControladorTP0(self)
        self.tp0.registrar(self.barra_lateral)
        self.barra_lateral.fijar_siempre_visible(SECCION_MENU)

        self.barra_lateral.grupo_actual = "TP1"
        self.tp1 = ControladorTP1(self)
        self.tp1.registrar(self.barra_lateral)

        self.barra_lateral.grupo_actual = "TP2"
        self.tp2 = ControladorTP2(self)
        self.tp2.registrar(self.barra_lateral)

        self.barra_lateral.grupo_actual = "TP3"
        self.tp3 = ControladorTP3(self)
        self.tp3.registrar(self.barra_lateral)

        self.barra_lateral.construir_selector("TP2")

    def establecer_original(self, imagen) -> None:
        self.imagen_original = imagen
        self.panel_original.establecer_imagen(imagen)

    def establecer_resultado(self, imagen) -> None:
        self.imagen_resultado = imagen
        self.panel_resultado.establecer_imagen(imagen)

    def elegir_ruta_apertura(self):
        return filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                (
                    "Imágenes",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pgm *.ppm *.raw "
                    "*.cr2 *.nef *.arw *.dng",
                ),
                ("Todos los archivos", "*.*"),
            ],
        )

    def elegir_ruta_guardado(self):
        return filedialog.asksaveasfilename(
            title="Guardar imagen resultado",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("Todos los archivos", "*.*")],
        )

    def mostrar_error(self, mensaje: str) -> None:
        messagebox.showerror("Error", mensaje)

    def mostrar_info(self, mensaje: str) -> None:
        messagebox.showinfo("Información", mensaje)

    def confirmar(self, mensaje: str) -> bool:
        return messagebox.askyesno("Confirmar", mensaje)

    def pedir_dimensiones_raw(self):
        texto_ancho = ctk.CTkInputDialog(
            title="Dimensiones de la imagen RAW",
            text="Este .raw no tiene encabezado (no es un RAW de cámara).\n"
                 "Ingresá el ANCHO en píxeles:",
        ).get_input()
        if not texto_ancho:
            return None

        texto_alto = ctk.CTkInputDialog(
            title="Dimensiones de la imagen RAW",
            text="Ingresá el ALTO en píxeles:",
        ).get_input()
        if not texto_alto:
            return None

        try:
            ancho, alto = int(texto_ancho), int(texto_alto)
        except ValueError:
            self.mostrar_error("Ancho y alto deben ser números enteros.")
            return None
        if ancho <= 0 or alto <= 0:
            self.mostrar_error("Ancho y alto deben ser mayores a 0.")
            return None
        return ancho, alto

    def _manejar_click_pixel(self, origen, x, y):
        imagen = self.imagen_original if origen == "original" else self.imagen_resultado
        if imagen is None:
            return
        valor = image_ops.obtener_pixel(imagen, x, y)
        self.panel_informacion.mostrar_pixel(origen, x, y, valor)
        for escucha in self.escuchas_click_pixel:
            escucha(origen, x, y, valor)

    def _manejar_seleccion_region(self, origen, x1, y1, x2, y2):
        imagen = self.imagen_original if origen == "original" else self.imagen_resultado
        if imagen is None:
            return
        try:
            estadisticas = image_ops.estadisticas_region(imagen, x1, y1, x2, y2)
        except ValueError:
            return
        self.ultima_seleccion = {"origen": origen, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        self.panel_informacion.mostrar_region(origen, estadisticas, (x1, y1, x2, y2))
        for escucha in self.escuchas_seleccion_region:
            escucha(origen, x1, y1, x2, y2)
