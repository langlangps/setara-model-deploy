from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import re
import string
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SETARA Model API")

# 1. Load Model & Tokenizer (Sekali saja saat API start)
MODEL_PATH = "./model/Model_IndoBERT_LampungUtara"
device = torch.device("cpu")  # Pakai CPU agar stabil di server/lokal

print("Loading IndoBERT model...")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


# 2. Schema Input
class BeritaRequest(BaseModel):
    judul: str


# 3. Helper Preprocessing
def clean_input(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[-+]?[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Atau spesifikkan domain frontend kamu
    allow_credentials=True,
    allow_methods=["*"],  # Ini akan mengizinkan OPTIONS, POST, dll.
    allow_headers=["*"],
)


# 4. Endpoint Prediksi
@app.post("/predict")
async def predict(request: BeritaRequest):
    try:
        text_clean = clean_input(request.judul)

        inputs = tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()

        label = True if prediction == 1 else False
        confidence = probs[0][prediction].item()

        return {
            "title": request.judul,
            "is_recommended": label,
            "confidence": round(confidence, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
