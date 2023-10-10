import cv2
import numpy as np

# Função para segmentar os morangos vermelhos
def segmentar_morangos(frame):
    
    # Converter o frame para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir a cor dos morangos
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = cv2.bitwise_or(mask1, mask2)

    # Aplicar operações morfológicas para melhorar a segmentação
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar os contornos dos morangos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar a contagem de morangos
    count = 0

    # Desenha os contornos
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Desenhar o círculo em torno do morango
                cv2.circle(frame, (cX, cY), 30, (0, 255, 0), 2)

                # Escrever "Morango" acima do contorno
                cv2.putText(frame, "Morango", (cX - 30, cY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Incrementar a contagem de morangos
            count += 1

    # Exibe a quantidade de morangos contados
    cv2.putText(frame, f"Morangos Maduros : {count}", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Abrir o vídeo
video = cv2.VideoCapture('Identificando-Morangos-com-Python/data/Morango.mp4')
if not video.isOpened():
    raise IOError("Erro ao abrir o vídeo.")

# Processar o frame do vídeo
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Chamar a função para segmentar os morangos
    frame_segmentado = segmentar_morangos(frame)

    # Exibir os morangos segmentados
    cv2.imshow("Segmentacao de Morangos", frame_segmentado)

    # Encerrar o código
    if cv2.waitKey(1) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
