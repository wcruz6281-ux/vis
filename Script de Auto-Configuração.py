import cv2
import pyttsx3
import numpy as np
import os
import urllib.request

# --- CONFIGURAÇÕES DE ARQUIVOS ---
PROTO = "deploy.prototxt"
MODEL = "mobilenet_iter_73000.caffemodel"

def inicializar_arquivos():
    # Se o prototxt não existir ou estiver incompleto, baixamos o oficial completo
    if not os.path.exists(PROTO):
        print("Baixando mapa da rede (prototxt)...")
        url_proto = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        urllib.request.urlretrieve(url_proto, PROTO)
    
    # Baixando o arquivo de pesos (o cérebro da IA)
    if not os.path.exists(MODEL):
        print("Baixando pesos da rede (caffemodel) - 23MB. Aguarde...")
        url_model = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
        urllib.request.urlretrieve(url_model, MODEL)
        print("Downloads concluídos!")

# --- CONFIGURAÇÃO DA IARA (VOZ) ---
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def iara_fala(texto):
    print(f"Iara: {texto}")
    engine.say(texto)
    engine.runAndWait()

# --- EXECUÇÃO ---
inicializar_arquivos()

try:
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
except Exception as e:
    print(f"Erro ao carregar os arquivos: {e}")
    print("Dica: Apague os arquivos .prototxt e .caffemodel da pasta e rode o script de novo.")
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

TRADUCAO = {"person": "pessoa", "car": "carro", "motorbike": "moto", "dog": "cachorro"}

cap = cv2.VideoCapture(0)
iara_fala("Sistema iniciado. Estou pronta para guiar você.")

while True:
    ret, frame = cap.read()
    if not ret: break
    (h, w) = frame.shape[:2]

    # Processamento da imagem
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.6:
            idx = int(detections[0, 0, i, 1])
            nome_en = CLASSES[idx]
            nome_pt = TRADUCAO.get(nome_en, nome_en)

            # Lógica de posição
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            centro_x = (startX + endX) / 2

            if centro_x < (w/3): direcao = "à esquerda"
            elif centro_x > (2*w/3): direcao = "à direita"
            else: direcao = "à frente"

            iara_fala(f"{nome_pt} {direcao}")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Iara Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()