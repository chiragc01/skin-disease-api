from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

# ── Allow requests from your website ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Disease info ──
CLASS_NAMES = {
    0: {
        "name": "Actinic Keratoses",
        "danger": "Needs attention",
        "color": "yellow",
        "what": "Rough, scaly patches caused by years of sun exposure.",
        "action": "Visit a skin doctor soon. It can turn into skin cancer if left untreated.",
        "common": "Common in people over 40 who spend a lot of time in the sun."
    },
    1: {
        "name": "Basal Cell Carcinoma",
        "danger": "Serious",
        "color": "red",
        "what": "A type of skin cancer that grows slowly on the outer layer of skin.",
        "action": "Please see a doctor as soon as possible.",
        "common": "Most common in people with fair skin."
    },
    2: {
        "name": "Benign Keratosis",
        "danger": "Not dangerous",
        "color": "green",
        "what": "Harmless skin growths. They are NOT cancer.",
        "action": "No urgent action needed. See a doctor if it bleeds or changes shape.",
        "common": "Very common in older adults. Usually brown or black."
    },
    3: {
        "name": "Dermatofibroma",
        "danger": "Not dangerous",
        "color": "green",
        "what": "A small, firm bump under the skin. Almost always harmless.",
        "action": "No treatment needed. See a doctor only if it hurts or grows bigger.",
        "common": "Often found on legs. More common in women."
    },
    4: {
        "name": "Melanoma",
        "danger": "Very serious",
        "color": "red",
        "what": "The most dangerous type of skin cancer. Can spread to other parts of the body.",
        "action": "Please see a doctor IMMEDIATELY. Early detection is very important.",
        "common": "Often looks like an unusual mole with uneven edges."
    },
    5: {
        "name": "Melanocytic Nevi",
        "danger": "Usually safe",
        "color": "green",
        "what": "Common moles. Almost everyone has them and they are usually harmless.",
        "action": "No action needed. See a doctor if the mole changes size or color.",
        "common": "Most common skin condition. Normal to have 10-40 moles."
    },
    6: {
        "name": "Vascular Lesions",
        "danger": "Usually harmless",
        "color": "yellow",
        "what": "Small marks caused by blood vessels near the surface of the skin.",
        "action": "No treatment needed. See a doctor if it bleeds or grows rapidly.",
        "common": "Common in all ages. Often looks like a small red or purple dot."
    },
}

# ── Load model once when server starts ──
print("Loading model...")
model = tf.keras.models.load_model("models/skin_disease_model.h5")
print("Model loaded!")

# ── Preprocess image ──
def preprocess(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype='float32') / 255.0
    return np.expand_dims(arr, axis=0)

# ── Health check endpoint ──
@app.get("/")
def root():
    return {"status": "Skin Disease API is running!"}

# ── Prediction endpoint ──
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Predict
    img_array = preprocess(image)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class]) * 100

    # Get all probabilities
    all_probs = [
        {
            "name": CLASS_NAMES[i]["name"],
            "probability": round(float(predictions[i]) * 100, 2)
        }
        for i in range(7)
    ]

    result = CLASS_NAMES[predicted_class]

    return {
        "predicted_class": predicted_class,
        "name": result["name"],
        "confidence": round(confidence, 2),
        "confidence_text": (
            "Very confident" if confidence >= 80 else
            "Fairly confident" if confidence >= 60 else
            "Not very confident" if confidence >= 40 else
            "Low confidence"
        ),
        "danger": result["danger"],
        "color": result["color"],
        "what": result["what"],
        "action": result["action"],
        "common": result["common"],
        "all_probabilities": all_probs
    }

