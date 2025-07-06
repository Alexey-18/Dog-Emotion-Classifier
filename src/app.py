#Backend
# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import json
import os
from PIL import Image

app = Flask(__name__)

# Загрузка модели
MODEL_PATH = "model/dog_emotion_classifier.h5"
CLASSES_PATH = "model/class_indices.json"

model = load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)

# Порядок классов
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file.stream).convert("RGB")
    img = img.resize((300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    pred_idx = np.argmax(preds[0])
    label = class_names[pred_idx]
    confidence = float(preds[0][pred_idx])

    return jsonify({"label": label, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
