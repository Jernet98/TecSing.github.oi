from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Configuración de MediaPipe para el seguimiento de las manos
deteccion_manoss = mp.solutions.hands
manos = deteccion_manoss.Hands(max_num_hands=1, min_detection_confidence=0.9, model_complexity=0, min_tracking_confidence=0.7)
dibujado = mp.solutions.drawing_utils

# Abrir la cámara
camara = cv2.VideoCapture(0)

def generar_frames():
    while True:
        ret, frame = camara.read()
        
        if not ret:
            print("¡No se ha podido registrar un frame!")
            break

        # Convertir el frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = manos.process(frame_rgb)

        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                dibujado.draw_landmarks(frame, hand_landmarks, deteccion_manoss.HAND_CONNECTIONS)

        # Codificar el frame como JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        
        # Convertir la imagen en bytes y devolverla
        frame_jpeg = jpeg.tobytes()
        
        # Enviar el frame a través del protocolo multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
