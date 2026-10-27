import re
import numpy as np
import cv2
import customtkinter as ctk

from app.core import image_io, image_ops

SECCION_MENU = "Menú"
SECCION_PIXEL = "Modificar Pixel"
SECCION_FUNCIONES = "Funciones:"

FUENTE_ENTRADA = ("Cascadia Code", 11)

class ControladorTP0:
    def __init__(self, app):
        self.app = app
        self.imagen_b = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.theta_1 = None
        self.theta_0 = None
        self.entrada_x = None
        self.entrada_y = None
        self.entrada_valor = None

        self.etiqueta_imagen_b = None

    def registrar(self, barra_lateral) -> None:
        self._construir_seccion_menu(barra_lateral)
        self._construir_seccion_pixel(barra_lateral)
        self._construir_seccion_funciones(barra_lateral)
        self.app.escuchas_click_pixel.append(self._al_hacer_click_pixel)

    def _construir_seccion_menu(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(
            SECCION_MENU, "Cargar imagen", self.cargar_imagen_original, primario=True
        )
        barra_lateral.agregar_boton(SECCION_MENU, "Guardar imagen", self.guardar_resultado)
        barra_lateral.agregar_boton(SECCION_MENU, "Copiar selección objeto", self.copiar_region_seleccionada_objeto)
        barra_lateral.agregar_boton(SECCION_MENU, "Copiar selección fondo", self.copiar_region_seleccionada_fondo)


    def cargar_imagen_original(self) -> None:
        ruta = self.app.elegir_ruta_apertura()
        if not ruta:
            return
        imagen = self._cargar_imagen_interactivo(ruta)
        if imagen is None:
            return

        self.app.establecer_original(imagen)
        self.app.establecer_resultado(imagen.copy())

    def _cargar_imagen_interactivo(self, ruta: str):
        try:
            return image_io.cargar_imagen(ruta)
        except image_io.DimensionesRawRequeridas:
            dimensiones = self.app.pedir_dimensiones_raw()
            if dimensiones is None:
                return None
            ancho, alto = dimensiones
            try:
                return image_io.cargar_raw_sin_encabezado(ruta, ancho, alto)
            except IOError as error:
                self.app.mostrar_error(str(error))
                return None
        except IOError as error:
            self.app.mostrar_error(str(error))
            return None

    def guardar_resultado(self) -> None:
        if self.app.imagen_resultado is None:
            self.app.mostrar_error("No hay imagen resultado para guardar.")
            return
        ruta = self.app.elegir_ruta_guardado()
        if not ruta:
            return
        try:
            image_io.guardar_imagen(ruta, self.app.imagen_resultado)
        except IOError as error:
            self.app.mostrar_error(str(error))
    def _obtener_region_seleccionada(self):
        seleccion = self.app.ultima_seleccion
        if seleccion is None:
            self.app.mostrar_error("Primero seleccioná una región arrastrando el mouse sobre una imagen.")
            return None

        imagen_origen = (
            self.app.imagen_original if seleccion["origen"] == "original" else self.app.imagen_resultado
        )
        if imagen_origen is None:
            self.app.mostrar_error("No hay imagen cargada en el panel donde se hizo la selección.")
            return None

        try:
            region = image_ops.copiar_region(
                imagen_origen, seleccion["x1"], seleccion["y1"], seleccion["x2"], seleccion["y2"]
            )
            return region
        except ValueError as error:
            self.app.mostrar_error(str(error))
            return None
    def copiar_region_seleccionada_objeto(self) -> None:
        seleccion = self.app.ultima_seleccion
        if seleccion is None:
            self.app.mostrar_error("Primero seleccioná una región arrastrando el mouse sobre una imagen.")
            return

        imagen_origen = (
            self.app.imagen_original if seleccion["origen"] == "original" else self.app.imagen_resultado
        )
        if imagen_origen is None:
            self.app.mostrar_error("No hay imagen cargada en el panel donde se hizo la selección.")
            return

        try:
            region = image_ops.copiar_region(
                imagen_origen, seleccion["x1"], seleccion["y1"], seleccion["x2"], seleccion["y2"]
            )
        except ValueError as error:
            self.app.mostrar_error(str(error))
            return

        # 1. CAPTURAR PUNTOS DE INICIO Y FIN (Asegurando que min/max funcionen independientemente de la dirección del arrastre)
        x_min = min(seleccion["x1"], seleccion["x2"])
        x_max = max(seleccion["x1"], seleccion["x2"])
        y_min = min(seleccion["y1"], seleccion["y2"])
        y_max = max(seleccion["y1"], seleccion["y2"])

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


        # Guardar las coordenadas de la región inicial (Bounding Box del objeto)
        self.bbox_objeto = {
            "x1": x_min,
            "y1": y_min,
            "x2": x_max,
            "y2": y_max,
            "ancho": x_max - x_min,
            "alto": y_max - y_min
        }

        # 2. CALCULAR EL PROMEDIO DE GRIS (theta_1)
        if region.ndim == 3:
            region_gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            region_gris = region

        self.theta_1 = float(np.mean(region_gris))

        # Imprimir en consola para verificar los valores capturados
        print(f"--- Objeto Registrado ---")
        print(f"Inicio (X1, Y1): ({x_min}, {y_min})")
        print(f"Fin    (X2, Y2): ({x_max}, {y_max})")
        print(f"Tamaño: {self.bbox_objeto['ancho']}x{self.bbox_objeto['alto']} px")
        print(f"Valor medio (theta_1): {self.theta_1:.2f}")

        # Mostrar la región recortada en la interfaz
        self.app.establecer_resultado(region)

    def copiar_region_seleccionada_fondo(self) -> None:
        region = self._obtener_region_seleccionada()
        if region is None:
            return

        if region.ndim == 3:
            region_gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            region_gris = region

        # Guardar valor medio del fondo
        self.theta_0 = float(np.mean(region_gris))
        print(f"Fondo registrado -> theta_0 (promedio): {self.theta_0:.2f}")

        # Mostrar en el panel de resultado
        self.app.establecer_resultado(region)

    def _construir_seccion_pixel(self, barra_lateral) -> None:
        marco = barra_lateral.agregar_seccion(SECCION_PIXEL)

        fila_coordenadas = ctk.CTkFrame(marco, fg_color="transparent")
        fila_coordenadas.pack(fill="x", pady=6)
        ctk.CTkLabel(fila_coordenadas, text="X:").pack(side="left", padx=(0, 4))
        self.entrada_x = ctk.CTkEntry(
            fila_coordenadas, width=50, font=FUENTE_ENTRADA, corner_radius=6, justify="center"
        )
        self.entrada_x.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(fila_coordenadas, text="Y:").pack(side="left", padx=(0, 4))
        self.entrada_y = ctk.CTkEntry(
            fila_coordenadas, width=50, font=FUENTE_ENTRADA, corner_radius=6, justify="center"
        )
        self.entrada_y.pack(side="left")

        fila_valor = ctk.CTkFrame(marco, fg_color="transparent")
        fila_valor.pack(fill="x", pady=6)
        ctk.CTkLabel(fila_valor, text="Valor:").pack(side="left", padx=(0, 4))
        self.entrada_valor = ctk.CTkEntry(
            fila_valor, placeholder_text="gris=128 o B,G,R",
            font=FUENTE_ENTRADA, corner_radius=6, justify="center",
        )
        self.entrada_valor.pack(side="left", fill="x", expand=True)

        barra_lateral.agregar_boton(SECCION_PIXEL, "Aplicar cambio", self.aplicar_cambio_pixel)

    def _al_hacer_click_pixel(self, origen, x, y, valor) -> None:
        if origen != "original":
            return
        self.entrada_x.delete(0, "end")
        self.entrada_x.insert(0, str(x))
        self.entrada_y.delete(0, "end")
        self.entrada_y.insert(0, str(y))
        self.entrada_valor.delete(0, "end")
        self.entrada_valor.insert(0, str(valor))

    def aplicar_cambio_pixel(self) -> None:
        if self.app.imagen_resultado is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        try:
            x = int(self.entrada_x.get())
            y = int(self.entrada_y.get())
        except ValueError:
            self.app.mostrar_error("X e Y deben ser números enteros.")
            return

        es_color = self.app.imagen_resultado.ndim == 3
        numeros = [int(n) for n in re.findall(r"-?\d+", self.entrada_valor.get())]
        if len(numeros) != (3 if es_color else 1):
            self.app.mostrar_error(
                "Valor inválido. Se esperan tres enteros B, G, R (ej: 10, 200, 30)."
                if es_color else "Valor inválido. Se espera un entero (ej: 128)."
            )
            return
        if any(n < 0 or n > 255 for n in numeros):
            self.app.mostrar_error("Los valores deben estar entre 0 y 255.")
            return
        valor = tuple(numeros) if es_color else numeros[0]

        try:
            actualizada = image_ops.establecer_pixel(self.app.imagen_resultado, x, y, valor)
        except IndexError as error:
            self.app.mostrar_error(str(error))
            return
        self.app.establecer_resultado(actualizada)

    def _construir_seccion_funciones(self, barra_lateral) -> None:
        marco = barra_lateral.agregar_seccion(SECCION_FUNCIONES)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Cargar imagen B", self.cargar_imagen_b)

        self.etiqueta_imagen_b = ctk.CTkLabel(
            marco, text="", text_color="gray70",
            font=("Roboto", 11), wraplength=220, justify="left",
        )
        self.etiqueta_imagen_b.pack(fill="x", pady=(0, 6))

        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Restar", self.restar_imagenes)

        ctk.CTkLabel(
            marco, text="Imagen original - Imagen B",
            text_color="gray70", font=("Roboto", 11),
        ).pack(fill="x", pady=(4, 4))

    def cargar_imagen_b(self) -> None:
        ruta = self.app.elegir_ruta_apertura()
        if not ruta:
            return
        imagen = self._cargar_imagen_interactivo(ruta)
        if imagen is None:
            return
        self.imagen_b = imagen
        nombre_archivo = ruta.replace("\\", "/").split("/")[-1]
        self.etiqueta_imagen_b.configure(text=f"Imagen B: {nombre_archivo}")

    def restar_imagenes(self) -> None:
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá la imagen original.")
            return
        if self.imagen_b is None:
            self.app.mostrar_error("Primero cargá la imagen B.")
            return

        original = self.app.imagen_original
        imagen_b = self.imagen_b
        if original.shape[:2] != imagen_b.shape[:2]:
            alto, ancho = original.shape[:2]
            alto_b, ancho_b = imagen_b.shape[:2]
            quiere_redimensionar = self.app.confirmar(
                f"La imagen B ({ancho_b}x{alto_b}) no tiene el mismo tamaño que "
                f"la original ({ancho}x{alto}).\n\n"
                "¿Querés redimensionar la imagen B para que coincida y poder restarlas?"
            )
            if not quiere_redimensionar:
                return
            imagen_b = image_ops.redimensionar_imagen(imagen_b, ancho, alto)

        try:
            diferencia = image_ops.restar_imagenes(original, imagen_b)
        except ValueError as error:
            self.app.mostrar_error(str(error))
            return
        self.app.establecer_resultado(diferencia)
