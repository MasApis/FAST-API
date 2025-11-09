from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("face_classifier.h5")
class_names = ["abdul", "farhan"]  # sesuaikan

@app.post("/predict")
async def predict(file: UploadFile):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).resize((150,150))

    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)
    label = class_names[np.argmax(pred)]

    return {"label": label}

