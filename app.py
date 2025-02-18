from flask import Flask, request, jsonify, send_file
from flasgger import Swagger
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
swagger = Swagger(app) 

model = YOLO("best.pt")  

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect():
    """
    Faz a detecção do drone em uma imagem usando YOLOv8.
    ---
    tags:
      - Deteção de Drones
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Imagem para detecção
    responses:
      200:
        description: Imagem processada com a detecção
        content:
          image/jpeg:
            schema:
              type: string
              format: binary
    """
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    image = cv2.imread(filename)

    results = model(filename)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caixa verde

    output_path = os.path.join(OUTPUT_FOLDER, "detected_" + file.filename)
    cv2.imwrite(output_path, image)

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
