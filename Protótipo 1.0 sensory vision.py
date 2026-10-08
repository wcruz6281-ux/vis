import cv2
import pyttsx3
import numpy as np
import sys
import pytesseract
from openai import OpenAI

client = OpenAI(
    api_key=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
    )
#  Voz
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def warn_usuario(texto):
    print("Alerta:", texto)
    engine.say(texto)
    engine.runAndWait()

#  IA
def explicar(texto):
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Explique de forma simples para uma pessoa cega."},
                {"role": "user", "content": texto}
            ]
        )
        return resposta.choices[0].message.content
    except:
        return "Erro ao explicar."

#  Controle leitura
ultimo_texto = ""
tempo_estavel = 0
frame_anterior = None

#  Modelo
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

TRADUCAO = {
    "car": "carro",
    "motorbike": "moto",
    "person": "pessoa"
}

try:
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', '')
except:
    sys.exit("Erro ao carregar modelo")

camera = cv2.VideoCapture(0)
warn_usuario("Sistema iniciado")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #  detectar se está parado
    if frame_anterior is not None:
        diff = cv2.absdiff(frame_anterior, gray)
        movimento = diff.mean()

        if movimento < 5:
            tempo_estavel += 1
        else:
            tempo_estavel = 0

    frame_anterior = gray

    #  objetos
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    deteccoes = net.forward()

    for i in range(0, deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]

        if confianca > 0.70:
            idx = int(deteccoes[0, 0, i, 1])
            nome_en = CLASSES[idx]
            nome_pt = TRADUCAO.get(nome_en, nome_en)

            warn_usuario(f"Detectei {nome_pt}")

    #  leitura

    if tempo_estavel > 20:
        texto = pytesseract.image_to_string(gray)

        if texto.strip() != "" and texto != ultimo_texto:
            warn_usuario("Lendo página")

            resposta = explicar(texto[:800])
            warn_usuario(resposta)

            ultimo_texto = texto
            tempo_estavel = 0

    cv2.imshow("Sensorial Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()