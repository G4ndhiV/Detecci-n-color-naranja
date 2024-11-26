import cv2
import numpy as np

def detect_oranges_with_gaussian_smooth(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de naranja en HSV
    lower_orange1 = np.array([0, 100, 100])
    upper_orange1 = np.array([10, 255, 255])
    
    lower_orange2 = np.array([11, 100, 100])
    upper_orange2 = np.array([20, 255, 255])
    
    lower_orange3 = np.array([21, 150, 100])
    upper_orange3 = np.array([30, 255, 255])
    
    # Crear máscaras
    mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
    mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
    mask3 = cv2.inRange(hsv, lower_orange3, upper_orange3)
    
    # Combinar máscaras
    combined_mask = cv2.bitwise_or(mask1, mask2)
    combined_mask = cv2.bitwise_or(combined_mask, mask3)
    
    # Suavizado Gaussiano
    smoothed_mask = cv2.GaussianBlur(combined_mask, (7, 7), 0)
    
    # Detectar contornos
    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos
    for contour in contours:
        epsilon = 0.007 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filtrar contornos grandes
        area = cv2.contourArea(contour)
        if area > 500:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
    
    return frame

# Capturar video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el frame. Saliendo ...")
        break

    # Llamar función para detectar naranja
    frame_with_gaussian_smooth = detect_oranges_with_gaussian_smooth(frame)

    # Mostrar resultado
    cv2.imshow('Detección de Naranja', frame_with_gaussian_smooth)

    # "q" para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
