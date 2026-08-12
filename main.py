from fastapi import FastAPI
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import os

app = FastAPI()

MODEL_NAME = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

@app.get("/")
def home():
    return {"status": "online"}

@app.get("/translate")
def translate(text: str, src: str, to: str):
    if not text or not text.strip() or src == to:
        return {"translated_text": text}
    try:
        tokenizer.src_lang = src
        encoded_input = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(to))
        risultato = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return {"translated_text": risultato[0]}
    except Exception as e:
        return {"translated_text": f"[Errore IA]: {str(e)}"}
