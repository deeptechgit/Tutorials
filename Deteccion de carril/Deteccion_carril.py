import numpy as np 
import cv2 as cv 

# Creamos la funcion que nos permite separar nuestra region de interes del resto de la imagen
def region_de_interes(img,vertices):
    # definimos una mascara en blanco
    mascara = np.zeros_like(img)

    # Definir un color de 3 canales o 1 canal para llenar la máscara.
    if len(img.shape)>2:
        cantidad_canales = img.shape[2]  # el indice puede ser 3 o 4 dependiento la imagen 
        mask_color_quitar = (255,) * cantidad_canales
    else:
        mask_color_quitar = 255
    #Rellenar píxeles dentro del polígono.

    cv.fillPoly(mascara, vertices, mask_color_quitar)

    # Construir la región de interés en función de dónde los píxeles de la máscara son distintos de cero.
    imagen_enmascarada = cv.bitwise_and(img, mascara)
    return imagen_enmascarada

# Creamos una función para graficar las lineas 
def draw_lines(img, lineas, color=[255,0,0],thickness=2):
    
    if lineas is not None:
        for linea in lineas:
            for x1,y1,x2,y2 in linea:
                cv.line(img,(x1,y1),(x2,y2), color, thickness)
                
# Creamos la funcion para el calculo de las lineas con hough
def hough_lineas(img, rho, theta, threshold, min_line_len, max_line_gap):
    lineas = cv.HoughLinesP(img, 
                            rho, 
                            theta, 
                            threshold, 
                            minLineLength = min_line_len, 
                            maxLineGap = max_line_gap)

    # Dibujamos todas las linea encontradas en la imagen  en una nueva imagen 
    img_hough = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(img_hough, lineas)
    return img_hough, lineas

# funcion para separar las lineas de izquierda y derecha del carril

def separando_izquierda_derecha(lineas):
    """ Separamos las lineas izquierda y derecha dependeindo de la pendiente"""
    lineas_izquierda = []
    lineas_derecha = []
    if lineas is not None:
        for linea in lineas:
            for x1,y1,x2,y2 in linea:
                # Si la cordenada y1 es mayor que y2 la pendiente es positiva y pertenece al lado izquierdo
                if y1>y2:
                    lineas_izquierda.append([x1,y1,x2,y2])
                # Si la cordenada y1 es menor que y2 la pendiente es negativa y pertenece al lado derecho
                elif y1<y2:
                    lineas_derecha.append([x1,y1,x2,y2])
                    
    return lineas_izquierda,lineas_derecha

# Funcion para calcular el promedio 
def cal_promedio(valores):
    """Calcula el valor promedio."""
    if not (type(valores) == 'NoneType'):
        if len(valores) > 0:
            n = len(valores)
        else:
            n = 1
        return sum(valores)/n
    
# Funcion para extrapòlar las lineas
def extrapolar_lineas(lineas, borde_superior, borde_inferior):
    """Extrapolar líneas teniendo en cuenta las intersecciones de los bordes inferior y superior"""
    pendientes = [] #pendientes de cada linea
    consts = []   # constantes de la formula de pendiente en cada linea 
    if lineas is not None:
        for x1,y1,x2,y2 in lineas:
            pendiente = (y1-y2)/(x1-x2)
            pendientes.append(pendiente)
            c = y1-pendiente*x1
            consts.append(c)
        
    prom_pendiente = cal_promedio(pendientes)
    prom_constantes = cal_promedio(consts)
    
    # Calcular la intersección promedio en el borde inferior
    x_punto_bajo_carril = int((borde_inferior - prom_constantes)/prom_pendiente)
    
    # Calcular la intersección promedio en el borde_superior
    x_punto_superior_carril = int((borde_superior - prom_constantes)/prom_pendiente)
    
    return [x_punto_bajo_carril, borde_inferior, x_punto_superior_carril, borde_superior]

def extrapolar_lineas_img(img, lineas, roi_borde_superior,roi_borde_inferior):
    # Crea una matriz en blanco para contener los resultados (coloreados)
    lineas_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Utilice la función definida anteriormente para identificar listas de líneas del lado izquierdo y derecho.
    lineas_izquierda, lineas_derecha = separando_izquierda_derecha(lineas)

    # Utilice la función definida anteriormente para extrapolar las listas de líneas a carriles reconocidos.
    linea_izquierda = extrapolar_lineas(lineas_izquierda, roi_borde_superior, roi_borde_inferior)
    linea_derecha = extrapolar_lineas(lineas_derecha, roi_borde_superior, roi_borde_inferior)
    
    if linea_izquierda is not None and linea_derecha is not None:
        draw_carril(lineas_img,[[linea_izquierda],[linea_derecha]])
        
    return lineas_img 

def draw_carril(img, lines):
    """ Completar el Area del carril."""
    points = []
    for x1,y1,x2,y2 in lines[0]:
        points.append([x1,y1])
        points.append([x2,y2])
    for x1,y1,x2,y2 in lines[1]:
        points.append([x2,y2])
        points.append([x1,y1])

    points = np.array([points], dtype = 'int32')        
    cv.fillPoly(img, points, (0,255,0))
    
def procesar_imagen(imagen):
    
    # convertimos a escala de grises
    img_gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    
    # Seleccionamos la intensidad para la Umbralizacion
    img_gray_selec = cv.inRange(img_gray, 150, 255)
    
    # Creamos la Mascara de nuestra region de interes
    vertices_roi = np.array([[[100,450],[900,540],[525,300],[440,330]]])
    img_gray_roi_selec = region_de_interes(img_gray_selec, vertices_roi)
    
    # Detecccion de Bordes con canny
    umbral_bajo = 50
    umbral_alto = 100
    img_canny = cv.Canny(img_gray_roi_selec, umbral_bajo, umbral_alto)
    
    # Remover el ruido usando gausian blur
    kernel_size = 5
    img_canny_blur = cv.GaussianBlur(img_canny,(kernel_size, kernel_size),0)
    
    # Parametros de la trasformada hoght de acuerdo a la imagen de entrada
    rho = 1
    theta = np.pi/180
    threshold = 100
    min_line_len = 50
    max_line_gap = 300
    
    hough, lineas = hough_lineas(img_canny_blur, rho, theta, threshold, min_line_len, max_line_gap)
    
    
    # Extrapolando las lineas
    roi_borde_superior = 330
    roi_borde_inferior = 540
    lineas_img = extrapolar_lineas_img(imagen,lineas, roi_borde_superior, roi_borde_inferior )
    
    #Combinamos dando pesos a la imagen
    imagen_resultante = cv.addWeighted(imagen, 1, lineas_img, 0.4, 0.0)
    return imagen_resultante

# Inicialisando la Captura de Video
video_cap = cv.VideoCapture("./carril.mp4")

if not video_cap.isOpened():
    print("Error en la lectura de video")
    
# Declarando parametros para guardar el video
frame_w = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(video_cap.get(cv.CAP_PROP_FPS))

# Declaramoe la codificacion fourcc para el archivo mp4
fourcc = cv.VideoWriter_fourcc(*"mp4v")

# declaramos el nombre del archivo de video
name_video_out = "carril_video_salida.mp4"

# Creamos el objeto de grabado de video
video_writer = cv.VideoWriter(name_video_out, fourcc, frame_fps, (frame_w, frame_h))

while video_cap.isOpened():
    ret, frame = video_cap.read()
    
    if ret:
        
        result = procesar_imagen(frame)
        cv.imshow("Salida", result)
        video_writer.write(result)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    	
    else:
        print("Fallo al leer")
        break
    
video_cap.release()
video_writer.release()
cv.destroyAllWindows()