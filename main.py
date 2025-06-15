from datetime import datetime
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import numpy as np
import csv, io
import tensorflow as tf  

import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("try-lokatani.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# Export Data Endpoint
@app.get("/export-data")
def export_data():
    docs = db.collection("db-scan-lokatani").stream()
    raw_data = [doc.to_dict() for doc in docs]

    if not raw_data:
        return {"message": "No data found"}

    field_map = {
        "user": "User",
        "vegResult": "Jenis Sayur",
        "vegWeight": "Berat Sayur (gram)",
        "date": "Waktu Input Data"
    }

    csv_data = [
        {field_map[key]: item.get(key, "") for key in field_map}
        for item in raw_data
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(field_map.values()), delimiter=';')
    writer.writeheader()
    writer.writerows(csv_data)

    output.seek(0)

    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"export-{today_str}.csv"

    return StreamingResponse(
        output,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# Upload Data Endpoint
class ScanData(BaseModel):
    user: str
    vegResult: str
    vegWeight: int
    date: datetime

@app.post("/upload-data")
async def upload_data(scans: List[ScanData]):
    for scan in scans:
        data = {
            "user": scan.user,
            "vegResult": scan.vegResult,
            "vegWeight": scan.vegWeight,
            "date": scan.date.isoformat()
        }
        db.collection("db-scan-lokatani").add(data)
    return {"message": "Data uploaded successfully"}


# Load H5 model
model = tf.keras.models.load_model("cnn_sayur_model.h5")

# Image Preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))  
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
    predicted_label = ["bayam merah", "kale"][predicted_index]  

    return {
        "class_label": predicted_label,
        "probabilities": prediction[0].tolist()
    }
