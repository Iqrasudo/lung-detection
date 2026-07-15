from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:8080"
            ]
        }
    }
)


# -----------------------------
# Configuration
# -----------------------------
CLASS_NAMES = ["covid19", "normal", "pneumonia"]

print("Loading TFLite model...")

interpreter = Interpreter(model_path="weights/model.tflite")
interpreter.allocate_tensors()

INPUT_DETAILS = interpreter.get_input_details()[0]
OUTPUT_DETAILS = interpreter.get_output_details()[0]

print("TFLite model loaded successfully!")
print("Input Details:", INPUT_DETAILS)
print("Output Details:", OUTPUT_DETAILS)

# MODEL_PATH = hf_hub_download(
#     repo_id="iqrakhawar/chest_xray_deep_learning",
#     filename="10-Final_chest_xray_AI_model.keras"
# )

# model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()

# with open("weights/model.tflite", "wb") as f:
#     f.write(tflite_model)

# print("done -> weights/model.tflite") 
# print("Loading model...")
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# print("Model loaded successfully!")
# print("Model input shape:", model.input_shape)
# -----------------------------
# Severity Map
# -----------------------------
severity_map = {

    "normal": {
        "severity": "No Disease",
        "risk": "Low",
        "follow_up": (
           "The chest X-ray does not show any significant signs of pneumonia or COVID-19 infection. "
            "Maintain a healthy lifestyle, stay hydrated, and continue practicing good respiratory hygiene. "
            "If symptoms such as persistent cough, fever, chest pain, or difficulty breathing develop or worsen, "
            "consult a healthcare professional for further clinical evaluation. Routine medical follow-up is "
            "recommended only if symptoms persist or new respiratory complaints arise."

        )
    },

    "pneumonia": {
        "severity": "Moderate",
        "risk": "Medium",
        "follow_up": (
                       "The chest X-ray findings are suggestive of pneumonia. It is recommended to consult a physician or "
            "pulmonologist as soon as possible for a detailed clinical assessment and confirmation of the diagnosis. "
            "Additional investigations such as blood tests, sputum culture, or repeat chest imaging may be required "
            "depending on the patient's condition. Follow the prescribed treatment plan, take medications exactly "
            "as directed, ensure adequate rest and hydration, and seek immediate medical attention if symptoms such "
            "as high fever, severe chest pain, worsening cough, or shortness of breath become more severe."

        )
    },

    "covid19": {
        "severity": "Critical",
        "risk": "High",
        "follow_up": (
           "The chest X-ray findings are highly suggestive of COVID-19-related lung involvement. Immediate medical "
            "evaluation is strongly recommended. The patient should promptly consult a qualified healthcare provider "
            "or visit the nearest hospital for further assessment and appropriate management. Additional laboratory "
            "tests, oxygen saturation monitoring, and confirmatory diagnostic testing may be required. If symptoms "
            "such as severe breathing difficulty, persistent chest pain, confusion, bluish lips or face, or low "
            "oxygen saturation are present, emergency medical care should be sought without delay. Follow all medical "
            "advice, isolation guidelines, and treatment recommendations provided by healthcare professionals."

        )
    }

}

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess(img):

    img = img.convert("L")          # grayscale
    img = img.resize((224, 224))

    img = np.array(img).astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "DeepLung AI Backend is Running"
    })


# -----------------------------
# Health Check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True
    })


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided."
        }), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({
            "error": f"Invalid image: {str(e)}"
        }), 400

    x = preprocess(img)

    interpreter.set_tensor(INPUT_DETAILS["index"], x)
    interpreter.invoke()
    probs = interpreter.get_tensor(OUTPUT_DETAILS["index"])[0]

    idx = int(np.argmax(probs))

    label = CLASS_NAMES[idx]

    confidence = float(probs[idx] * 100)

    info = severity_map[label]

    return jsonify({

        "prediction": label,

        "confidence": round(confidence, 2),

        "estimated_severity": info["severity"],

        "risk_level": info["risk"],

        "follow_up": info["follow_up"],

        "probabilities": {

            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)

            for i in range(len(CLASS_NAMES))
        }

    })


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )