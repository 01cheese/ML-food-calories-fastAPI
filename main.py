from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import json
import io

app = FastAPI()

# CORS
origins = ["http://localhost:63342", "http://127.0.0.1:63342"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/styles", StaticFiles(directory="styles"), name="styles")

# Модель и данные
processor = AutoProcessor.from_pretrained("nateraw/food")
model = AutoModelForImageClassification.from_pretrained("nateraw/food")
labels = model.config.id2label

with open("food101.json", "r") as f:
    nutrition_data = json.load(f)

# Роуты
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict_food(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
    label = labels[idx]
    nutrition = nutrition_data.get(label, {})
    return JSONResponse({
        "predicted_label": label,
        "nutrition": nutrition
    })
