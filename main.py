import os
import re
import string
import torch
from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, field_validator
from transformers import BertForSequenceClassification, BertTokenizer
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from dotenv import load_dotenv

app = FastAPI(title="SETARA Model API")

# --- KONFIGURASI KEAMANAN (API KEY) ---
load_dotenv()
# Anda bisa mengatur variabel lingkungan SETARA_MODEL_API_KEY di server
# Jika tidak diatur, default-nya adalah 'KunciRahasiaSetara2024'
API_KEY = os.getenv("SETARA_MODEL_API_KEY")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
async def validate_api_key(header_api_key: str = Security(api_key_header)):
    """
    Fungsi untuk memvalidasi apakah API Key yang dikirim di header cocok.
    """
    if header_api_key == API_KEY:
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, 
        detail="Akses Ditolak: API Key tidak valid atau tidak ditemukan di header X-API-KEY"
    )

# --- LOAD MODEL & TOKENIZER ---
# Pastikan path model sesuai dengan lokasi di server Anda
MODEL_PATH = "./model/Model_IndoBERT_LampungUtara"
device = torch.device("cpu")

print("Sedang memuat model IndoBERT...")
try:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")

# --- SCHEMA INPUT ---
class BeritaRequest(BaseModel):
    # Mengizinkan input berupa string tunggal atau list of strings
    judul: Union[str, List[str]]

    @field_validator('judul')
    @classmethod
    def validate_judul(cls, v):
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError('List judul tidak boleh kosong')
            for i, item in enumerate(v):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f'Item pada indeks {i} harus berupa string yang tidak kosong')
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError('Judul tidak boleh kosong')
        return v

# --- HELPER PREPROCESSING ---
def clean_input(text: str):
    """
    Membersihkan teks input sebelum dimasukkan ke model.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[-+]?[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CORE LOGIC PREDIKSI ---
def batch_predict(titles: List[str]) -> List[Dict[str, Any]]:
    """
    Menjalankan inferensi model secara batch untuk efisiensi.
    """
    cleaned_titles = [clean_input(t) for t in titles]
    
    # Tokenisasi secara batch
    inputs = tokenizer(
        cleaned_titles,
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
        predictions = torch.argmax(logits, dim=-1)

    results = []
    for i, title in enumerate(titles):
        pred_idx = predictions[i].item()
        results.append({
            "title": title,
            "is_recommended": True if pred_idx == 1 else False,
            "confidence": round(probs[i][pred_idx].item(), 4),
        })
    return results

# --- ENDPOINT PREDIKSI ---
@app.post("/predict", dependencies=[Depends(validate_api_key)])
async def predict(request: BeritaRequest):
    """
    Endpoint utama untuk prediksi berita.
    Memerlukan header 'X-API-KEY'.
    """
    try:
        # Cek apakah input tunggal atau list
        is_single = isinstance(request.judul, str)
        titles_to_process = [request.judul] if is_single else request.judul

        # Proses batch prediction
        predictions = batch_predict(titles_to_process)

        # Jika input awal adalah string tunggal, kembalikan objek tunggal
        if is_single:
            return predictions[0]
        
        # Jika input awal adalah list, kembalikan dalam format list
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    # Menjalankan pada host 0.0.0.0 agar bisa diakses secara eksternal
    uvicorn.run(app, host="0.0.0.0", port=8001)