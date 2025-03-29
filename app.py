# @author: Carlos da Costa
# @version: 1.0
# @date: 29/03/2025
# @tutorial/documentation: https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html

# Importação do OpenCV.
import cv2


# Variável de inicialização da webcam. 0 indica que a webcam padrão do computador.
video = cv2.VideoCapture(0)

# Variável de carregamento do classificador pré-treinado 'Haar Cascade' para rosto.
cascade_rosto = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
    # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo.
    ret, frame = video.read()

    # Loop de Captura de Vídeo. Se ret estivar disponível, o algoritmo prossegue.
    # Mantém a captura ativa até que o usuário pressione a tecla 'q'.
    if not ret:
        break

    # Converte a imagem para tons de cinza melhora o desempenho da detecção.   
    escala_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale(): Detecta rostos na imagem.
    # scaleFactor=1.1: Reduz gradualmente o tamanho da imagem para detectar rostos em diferentes escalas.
    # minNeighbors=5: Controla a quantidade de vizinhos para eliminar falsos positivos.
    # minSize=(30, 30): Define o tamanho mínimo do rosto a ser detectado.
    rostos = cascade_rosto.detectMultiScale(escala_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar retângulos ao redor dos rostos detectados.
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe o vídeo com detecção
    cv2.imshow("Detecção em Tempo Real", frame)

    # cv2.waitKey(1): Aguarda 1 milissegundo por uma tecla pressionada.
    # ord('q'): Se a tecla 'q' for pressionada, o loop termina.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam.
video.release()

# Fecha todas as janelas abertas pelo OpenCV.
cv2.destroyAllWindows()