from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load H5 model
model = tf.keras.models.load_model("cnn_sayur_model.h5")

# Image Preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))  # Sesuaikan dengan input model kamu
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict Endpoint
@app.post("/cnn")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)

    prediction = model.predict(input_data)
    predicted_index = int(np.argmax(prediction[0]))
    predicted_label = ["bayam merah", "kale"][predicted_index]  # Ganti sesuai label kamu

    return JSONResponse({
        "class_label": predicted_label,
        "probabilities": prediction[0].tolist()
    })

@app.get("/")
def home():
    return{"status": "Server ready"}
