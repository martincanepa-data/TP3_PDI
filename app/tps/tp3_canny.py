import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import cv2
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt

from app.tps.tp2_filtros import ControladorTP2
from app.tps.tp1_operadores import ControladorTP1

SECCION_BORDES = "Bordes Avanzados"
MASCARA_SOBEL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=float)

def susan_mask():
    mask=np.ones((7,7))
    mask[0,0]=0
    mask[0,1]=0
    mask[0,5]=0
    mask[0,6]=0
    mask[1,0]=0
    mask[1,6]=0
    mask[5,0]=0
    mask[5,6]=0
    mask[6,0]=0
    mask[6,1]=0
    mask[6,5]=0
    mask[6,6]=0
    return mask
class ControladorTP3(ControladorTP2):
    def __init__(self, app):
        super().__init__(app)
        self.mascara_inicial= None

    def registrar(self, barra_lateral) -> None:
        self._construir_seccion_bordes_avanzados(barra_lateral)
        #self._construir_seccion_filtros(barra_lateral)


    def _construir_seccion_bordes_avanzados(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_BORDES, "Canny", self.aplicar_canny)
        barra_lateral.agregar_boton(SECCION_BORDES, "SUSAN", self.aplicar_susan)
        barra_lateral.agregar_boton(SECCION_BORDES, "Transformada de Hough", self.aplicar_transformada_hough)
        barra_lateral.agregar_boton(SECCION_BORDES, "Contornos Activos", self.aplicar_intercambio_pixeles)
        barra_lateral.agregar_boton(SECCION_BORDES, "Seguir Objeto", self.aplicar_seguir_objeto)

    def aplicar_canny(self):
        if self.app.imagen_original is None:
            ctk.messagebox.showerror("Error", "No hay imagen cargada.")
            return
        imagen = self.app.imagen_original

        if imagen.ndim == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        # Aplicar el detector de bordes Canny
        
        # Primero hay que pasarle el filtro de gaussiano para reducir el ruido
        controlador = ControladorTP1(self.app)

        sigma = 2.0
        tamano = round(2 * sigma + 1)

        mascara = controlador.mascara_gauss(tamano, sigma)

        def promedio_pesado(ventana):
            return np.sum(ventana * mascara)

        imagen_suavizada = controlador.ventana_deslizante(imagen_gris,tamano,promedio_pesado)

        # Sobel X
        Ix = self.ventana_deslizante(imagen_suavizada,3,lambda v: np.sum(v * MASCARA_SOBEL))
        
        # Sobel Y
        Iy = self.ventana_deslizante(imagen_suavizada,3,lambda v: np.sum(v * MASCARA_SOBEL.T))
        
        magnitud = np.sqrt(Ix**2 + Iy**2)
        angulo = np.degrees(np.arctan2(Iy, Ix))

        def asignar_direccion(angulo):
            alto, ancho = angulo.shape
            direccion = np.zeros((alto, ancho), dtype=np.uint8)
            for i in range(alto):
                for j in range(ancho):
                    a = angulo[i, j]

                    if a < 0:
                        a += 180

                    if (0 <= a < 22.5) or (157.5 <= a <= 180):
                        direccion[i, j] = 0

                    elif 22.5 <= a < 67.5:
                        direccion[i, j] = 45

                    elif 67.5 <= a < 112.5:
                        direccion[i, j] = 90

                    else:
                        direccion[i, j] = 135
            return direccion
        direccion = asignar_direccion(angulo)

        def eliminar_no_maximos(magnitud, direccion):
            alto, ancho = magnitud.shape
            resultado = np.zeros((alto, ancho), dtype=np.float32)

            for i in range(1, alto - 1):
                for j in range(1, ancho - 1):
                    angulo = direccion[i, j]

                    if angulo == 0:
                        vecinos = [magnitud[i, j - 1], magnitud[i, j + 1]]
                    elif angulo == 45:
                        vecinos = [magnitud[i - 1, j - 1], magnitud[i + 1, j + 1]]
                    elif angulo == 90:
                        vecinos = [magnitud[i - 1, j], magnitud[i + 1, j]]
                    else:  # angulo == 135
                        vecinos = [magnitud[i - 1, j + 1], magnitud[i + 1, j - 1]]

                    if max(vecinos) >= magnitud[i, j]:
                        resultado[i, j] = 0
                    else:
                        resultado[i, j] = magnitud[i, j]

            return resultado
        
        resultado=eliminar_no_maximos(magnitud,direccion)

        def histeresis(magnitud,t1,t2):


            alto, ancho = magnitud.shape

            resultado = np.zeros((alto, ancho), dtype=np.uint8)

            # Bordes fuertes
            resultado[magnitud > t2] = 255

            # Bordes débiles
            debiles = (magnitud >= t1) & (magnitud <= t2)

            for i in range(1, alto - 1):
                for j in range(1, ancho - 1):

                    if debiles[i, j]:

                        vecinos = resultado[i-1:i+2, j-1:j+2]

                        if np.any(vecinos == 255):
                            resultado[i, j] = 255

            return resultado
        
        t1 = self.pedir_umbral("Umbral inferior","Debe ser > 0 y < t2 ")
        t2 = self.pedir_umbral("Umbral superior","Debe ser > t1")

        histeresis_resultado = histeresis(resultado,t1,t2)

        self.app.establecer_resultado(histeresis_resultado)

        return histeresis_resultado

    def aplicar_susan(self):
        if self.app.imagen_original is None:
            ctk.messagebox.showerror("Error", "No hay imagen cargada.")
            return
        imagen = self.app.imagen_original

        if imagen.ndim == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()

        # Aplicar el detector de bordes SUSAN
        

        def susan_corner_detection(img, umbral=40, bordes=False):
            img = img.astype(np.float64)
            mascara_circular = susan_mask()
            output = np.zeros(img.shape)

            for i in range(3, img.shape[0]-3):
                for j in range(3, img.shape[1]-3):
                    ir = np.array(img[i-3:i+4, j-3:j+4])
                    ir = ir[mascara_circular == 1]
                    ir0 = img[i, j]
                    diferencias = np.abs(ir - ir0)
                    cantidad = np.sum(diferencias <= umbral)
                    s = 1 - (cantidad / 37)

                    if bordes:
                        if 0.35 <= s < 0.65:
                            output[i, j] = 1 
                    else:
                        if s >= 0.5:
                            output[i, j] = 1

            return output
        
        # Pintar los bordes detectados
        resultado = susan_corner_detection(imagen_gris,bordes=False)
        imagen_resultado = cv.cvtColor(imagen_gris, cv.COLOR_GRAY2BGR)
        imagen_resultado[resultado != 0] = [0, 0, 255]

        self.app.establecer_resultado(imagen_resultado)


    def aplicar_transformada_hough(self):

        bordes = self.aplicar_canny()

        def hough_rectas(bordes):
            alto, ancho = bordes.shape

            # La diagonal máxima posible
            diagonal = int(np.ceil(np.sqrt(alto**2 + ancho**2)))
            
            # -diagonal a +diagonal abarca una longitud de 2 * diagonal
            num_rhos = 2 * diagonal
            rhos = np.linspace(-diagonal, diagonal, num_rhos)
            thetas = np.deg2rad(np.arange(-90, 90))

            acumulador = np.zeros((num_rhos, len(thetas)), dtype=np.uint64)

            # Precalcular senos y cosenos para los ángulos (evita recalcular en cada punto)
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)

            y_idxs, x_idxs = np.nonzero(bordes)

            # Vectorización sobre los ángulos theta para cada pixel de borde
            for x, y in zip(x_idxs, y_idxs):
                # Proyección rho para todos los ángulos theta
                rho_vals = x * cos_t + y * sin_t
                
                # Convertir el valor de rho continuo al índice del acumulador [0, num_rhos - 1]
                rho_idxs = np.round(rho_vals + diagonal).astype(int)
                
                # Asegurar que el índice no supere los límites del arreglo
                rho_idxs = np.clip(rho_idxs, 0, num_rhos - 1)

                # Acumular
                for j, rho_idx in enumerate(rho_idxs):
                    acumulador[rho_idx, j] += 1

            return acumulador, rhos, thetas

        acumulador, rhos, thetas = hough_rectas(bordes)
        # Crear copia BGR para dibujar las líneas en color sobre la imagen original

        if len(self.app.imagen_original.shape) == 2:
            imagen_resultado = cv2.cvtColor(self.app.imagen_original, cv2.COLOR_GRAY2BGR)
        else:
            imagen_resultado = self.imagen_original.copy()

        umbral = self.pedir_umbral("Umbral para las rectas de Hough")

        filas, columnas = np.where(acumulador >= umbral)

        for i, j in zip(filas, columnas):
            rho = rhos[i]
            theta = thetas[j]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Dibujar línea (BGR: Verde = (0, 255, 0) / Rojo = (0, 0, 255))
            cv2.line(imagen_resultado, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Si tu interfaz usa RGB (ej. PIL / Tkinter / PyQt), convertila primero:
        imagen_resultado = cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB)

        # Enviar a la interfaz
        self.app.establecer_resultado(imagen_resultado)
    
    def pedir_umbral(self,titulo,texto):
        umbral = self._pedir_numero(titulo,texto)
        if umbral is None:
            return 
        if umbral <= 0:
            self.app.mostrar_error("El umbral debe ser mayor a 0.")
            return
        return umbral


    def aplicar_intercambio_pixeles(self):

        imagen_original = self.app.imagen_original

        if len(imagen_original.shape) == 3:
            img_gray = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            img_gray = imagen_original.astype(float)

        self.intercambio_pixel(img_gray,imagen_original)

    def intercambio_pixel(self,imagen_gris,imagen_original):   
        controlador_tp0 = self.app.tp0
        theta_1 = controlador_tp0.theta_1  # Promedio del Objeto
        theta_0 = controlador_tp0.theta_0  # Promedio del Fondo
        print('entramos a intercambio de pixel')
        print(theta_0)
        print(theta_1)

        i_inicial = controlador_tp0.x_min
        i_final = controlador_tp0.x_max      
        j_inicial = controlador_tp0.y_min
        j_final = controlador_tp0.y_max

        mascara = self.coloco_mascara(imagen_gris,i_inicial,i_final,j_inicial,j_final,theta_0,theta_1)
        mascara[mascara == 0] = 3
        v_Lin,mascara = self.obtener_lin(mascara)
        v_Lout,mascara = self.obtener_Lout(mascara,v_Lin)
        print("Lin inicial:", len(v_Lin))
        print("Lout inicial:", len(v_Lout))
        print("Valores mascara:", np.unique(mascara))
        puntoDcorte = 0

        while puntoDcorte != 500:

            # --- FASE 1: EXPANSIÓN (Crecimiento del objeto) ---
            # Evaluamos la frontera externa (Lout)
            lout_a_eliminar = set()
            lin_a_agregar = set()

            for (f, c) in list(v_Lout):  # list() evita errores al modificar el set
                val_pixel = float(imagen_gris[f, c])
                dist_0 = abs(theta_0 - val_pixel) + 1e-5
                dist_1 = abs(theta_1 - val_pixel) + 1e-5
                fd = math.log(dist_0 / dist_1)

                if fd > 0:  # El píxel se parece más al objeto
                    mascara[f, c] = -1  # Se convierte en frontera interna
                    lout_a_eliminar.add((f, c))
                    lin_a_agregar.add((f, c))

            # Aplicar los cambios de la Fase 1
            v_Lout.difference_update(lout_a_eliminar)
            v_Lin.update(lin_a_agregar)
            # --- FASE 2: CONTRACCIÓN (Achicamiento del objeto) ---
            # Evaluamos la frontera interna (Lin)
            lin_a_eliminar = set()
            lout_a_agregar = set()

            for (f, c) in list(v_Lin):
                val_pixel = float(imagen_gris[f, c])
                dist_0 = abs(theta_0 - val_pixel) + 1e-5
                dist_1 = abs(theta_1 - val_pixel) + 1e-5
                fd = math.log(dist_0 / dist_1)

                if fd < 0:  # El píxel se parece más al fondo
                    mascara[f, c] = 1  # Se convierte en frontera externa
                    lin_a_eliminar.add((f, c))
                    lout_a_agregar.add((f, c))

            # Aplicar los cambios de la Fase 2
            v_Lin.difference_update(lin_a_eliminar)
            v_Lout.update(lout_a_agregar)

            v_Lin, v_Lout, mascara = self.sanear_bordes(mascara, v_Lin, v_Lout)
            
            
            
            if puntoDcorte % 20 == 0:

                img_color = imagen_original.copy().astype(np.uint8)
     
                
                # Pintar el contorno actual de ROJO (RGB: 255, 0, 0)
                img_color[mascara == -1] = [0, 255, 0]
                img_color[mascara == 1] = [0, 0, 255]
                
                # Actualizar la interfaz
                self.app.establecer_resultado(img_color)
                self.app.update()  # Clave para que se vea la animación

            puntoDcorte+=1

        # Pintamos el contorno final de rojo y azul
        resultado_final = imagen_original.copy().astype(np.uint8)

        resultado_final[mascara == -1] = [255, 0, 0]
        resultado_final[mascara == 1] = [0, 0, 255]

        print(np.sum(mascara == -1))
        print(np.sum(mascara == 1))
        self.mascara_inicial = mascara
        self.app.establecer_resultado(resultado_final)

        
    def coloco_mascara(self,imagen,i_i,i_f,j_i,j_f,theta0,theta1):
        salida = np.zeros(imagen.shape)
        for fila in range(j_i,j_f):
            for columna in range(i_i,i_f):
                val_pixel = imagen[fila,columna]
                dist_fondo = abs(theta0 - val_pixel) + 1e-5
                dist_objeto = abs(theta1 - val_pixel) + 1e-5
                res = math.log(dist_fondo/dist_objeto)
                if res > 0:
                    salida[fila,columna] = -3 #pertenece al objeto
                if res < 0:
                    salida[fila,columna] = 3 #pertenece al fondo
        return salida
    def obtener_lin(self, mascara):
        alto, ancho = mascara.shape
        lin = set()  # Usamos un conjunto (set) para evitar duplicados y acelerar búsquedas

        # Desplazamientos de los 4 vecinos ortogonales (arriba, abajo, izquierda, derecha)
        vecinos_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for fila in range(alto):
            for columna in range(ancho):
                # Solo nos interesan los píxeles marcados como objeto (-3)
                if mascara[fila, columna] == -3:
                    # Verificar si alguno de sus 4 vecinos es fondo (3)
                    es_borde_interno = False
                    for df, dc in vecinos_4:
                        nf, nc = fila + df, columna + dc

                        # Verificar que el vecino esté dentro de los límites de la imagen
                        if 0 <= nf < alto and 0 <= nc < ancho:
                            if mascara[nf, nc] == 3:  # Vecino en el fondo
                                es_borde_interno = True
                                break

                    # Si linda con el fondo, se convierte en Lin (-1)
                    if es_borde_interno:
                        mascara[fila, columna] = -1
                        lin.add((fila, columna))

        return lin, mascara

    def obtener_Lout(self,mascara,Lin):
        alto, ancho = mascara.shape
        lout = set()

        # Los 4 vecinos ortogonales: Norte, Sur, Oeste, Este
        vecinos_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Recorremos SOLO las coordenadas almacenadas en Lin
        for fila, columna in Lin:
            for df, dc in vecinos_4:
                nf = fila + df       # Desplazamiento en Filas (N/S)
                nc = columna + dc    # Desplazamiento en Columnas (O/E)

                # Verificar que el vecino esté dentro del tamaño de la imagen
                if 0 <= nf < alto and 0 <= nc < ancho:
                    # Si el vecino pertenece al fondo lejano (3), pasa a ser Lout (1)
                    if mascara[nf, nc] == 3:
                        mascara[nf, nc] = 1
                        lout.add((nf, nc))

        return lout, mascara













        #    x_min = min(seleccion["x1"], seleccion["x2"])
        #    x_max = max(seleccion["x1"], seleccion["x2"])
        #    y_min = min(seleccion["y1"], seleccion["y2"])
        #    y_max = ma
   
    def sanear_bordes(self, mascara, v_Lin, v_Lout):
        alto, ancho = mascara.shape
        vecinos_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # --- PARTE 1: RECLUTAR NUEVOS VECINOS ---
        
        # 1A: Nuevos Lout (Píxeles de fondo lejano que ahora tocan Lin)
        nuevos_lout = set()
        for f, c in v_Lin:
            for df, dc in vecinos_4:
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho:
                    if mascara[nf, nc] == 3:  # Era fondo lejano
                        mascara[nf, nc] = 1   # Ahora es frontera externa
                        nuevos_lout.add((nf, nc))
        v_Lout.update(nuevos_lout)

        # 1B: Nuevos Lin (Píxeles de objeto profundo que ahora tocan Lout)
        nuevos_lin = set()
        for f, c in v_Lout:
            for df, dc in vecinos_4:
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho:
                    if mascara[nf, nc] == -3: # Era objeto profundo
                        mascara[nf, nc] = -1  # Ahora es frontera interna
                        nuevos_lin.add((nf, nc))
        v_Lin.update(nuevos_lin)


        # --- PARTE 2: ELIMINAR PÍXELES QUE YA NO SON FRONTERA ---
        
        # 2A: Limpiar Lin (Píxeles de Lin que quedaron enterrados adentro del objeto)
        lin_a_eliminar = set()
        for f, c in v_Lin:
            es_frontera = False
            for df, dc in vecinos_4:
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho:
                    if mascara[nf, nc] > 0: # Si toca a un Lout (1) o al Fondo (3)
                        es_frontera = True
                        break
            
            # Si solo está rodeado de píxeles negativos (-1 o -3), ya no es frontera
            if not es_frontera:
                mascara[f, c] = -3 # Pasa a ser objeto profundo
                lin_a_eliminar.add((f, c))
                
        v_Lin.difference_update(lin_a_eliminar)

        # 2B: Limpiar Lout (Píxeles de Lout que quedaron sueltos en el fondo)
        lout_a_eliminar = set()
        for f, c in v_Lout:
            es_frontera = False
            for df, dc in vecinos_4:
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho:
                    if mascara[nf, nc] < 0: # Si toca a un Lin (-1) o al Objeto (-3)
                        es_frontera = True
                        break
            
            # Si solo está rodeado de píxeles positivos (1 o 3), ya no es frontera
            if not es_frontera:
                mascara[f, c] = 3 # Pasa a ser fondo lejano
                lout_a_eliminar.add((f, c))
                
        v_Lout.difference_update(lout_a_eliminar)
        
        # Devolvemos las listas y la máscara actualizadas
        return v_Lin, v_Lout, mascara
    
    
    def aplicar_seguir_objeto(self):
        controlador_tp0 = self.app.tp0
        img1 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a1.jpg")
        img2 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a2.jpg")
        img3 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a3.jpg")
        img4 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a4.jpg")
        img5 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a5.jpg")
        img6 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a6.jpg")
        img7 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a7.jpg")
        img8 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a8.jpg")
        img9 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a9.jpg")
        img10 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a10.jpg")
        img11 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a11.jpg")
        img12 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a12.jpg")
        img13 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a13.jpg")
        img14 = cv2.imread(r"C:\Users\skate\Documentos\Documents\PDI\TP2-PDI-\imagenes\a14.jpg")
        mascara = self.mascara_inicial

        imagenes = [img1, img2, img3, img4,img5, img6, img7, img8 ,img9, img10, img11, img12, img13, img14]
        theta_1 = controlador_tp0.theta_1  # Promedio del Objeto
        theta_0 = controlador_tp0.theta_0  # Promedio del Fondo 
        self.seguir_objeto(imagenes,mascara,theta_0,theta_1)


    def seguir_objeto(self, lista_imagenes, mascara_inicial, theta_0, theta_1):

        mascara = mascara_inicial

        for imagen_original in lista_imagenes:

            if len(imagen_original.shape) == 3:
                imagen_gris = cv2.cvtColor(
                    imagen_original,
                    cv2.COLOR_BGR2GRAY
                ).astype(float)
            else:
                imagen_gris = imagen_original.astype(float)
            

            v_Lin  = self.reconstruir_lin(mascara)
            v_Lout  = self.reconstruir_lout(mascara)
            print("Lin:", len(v_Lin))
            print("Lout:", len(v_Lout))
            for iteracion in range(30):

                # -----------------
                # EXPANSION
                # -----------------

                lout_a_eliminar = set()
                lin_a_agregar = set()

                for (f, c) in list(v_Lout):

                    val_pixel = float(imagen_gris[f, c])

                    dist_0 = abs(theta_0 - val_pixel) + 1e-5
                    dist_1 = abs(theta_1 - val_pixel) + 1e-5

                    fd = math.log(dist_0 / dist_1)

                    if fd > 0:

                        mascara[f, c] = -1

                        lout_a_eliminar.add((f, c))
                        lin_a_agregar.add((f, c))

                v_Lout.difference_update(lout_a_eliminar)
                v_Lin.update(lin_a_agregar)

                # -----------------
                # CONTRACCION
                # -----------------

                lin_a_eliminar = set()
                lout_a_agregar = set()

                for (f, c) in list(v_Lin):

                    val_pixel = float(imagen_gris[f, c])

                    dist_0 = abs(theta_0 - val_pixel) + 1e-5
                    dist_1 = abs(theta_1 - val_pixel) + 1e-5

                    fd = math.log(dist_0 / dist_1)

                    if fd < 0:

                        mascara[f, c] = 1

                        lin_a_eliminar.add((f, c))
                        lout_a_agregar.add((f, c))

                v_Lin.difference_update(lin_a_eliminar)
                v_Lout.update(lout_a_agregar)

                v_Lin, v_Lout, mascara = self.sanear_bordes(
                    mascara,
                    v_Lin,
                    v_Lout
                )

                if iteracion % 5 == 0:

                    vista = imagen_original.copy()

                    vista[mascara == -1] = [255, 0, 0]
                    vista[mascara == 1] = [0, 0, 255]

                    self.app.establecer_resultado(vista)
                    self.app.update()

            resultado = imagen_original.copy()

            resultado[mascara == -1] = [255, 0, 0]
            resultado[mascara == 1] = [0, 0, 255]

            self.app.establecer_resultado(resultado)
            self.app.update()

        return mascara

    def reconstruir_lin(self, mascara):

        lin = set()

        filas, columnas = np.where(mascara == -1)

        for f, c in zip(filas, columnas):
            lin.add((f, c))

        return lin
    def reconstruir_lout(self, mascara):

        lout = set()

        filas, columnas = np.where(mascara == 1)

        for f, c in zip(filas, columnas):
            lout.add((f, c))

        return lout