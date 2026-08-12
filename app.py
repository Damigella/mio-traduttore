import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

@st.cache_resource # Evita di ricaricare il modello a ogni click, risparmiando memoria
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

LANG_CODES = {"Italiano": "it", "Inglese": "en", "Spagnolo": "es", "Francese": "fr"}

st.title("🌍 Il Mio Traduttore Indipendente")

testo = st.text_area("Testo da tradurre", placeholder="Scrivi qui...")
col1, col2 = st.columns(2)
with col1:
    lingua_orig = st.selectbox("Lingua di Partenza", list(LANG_CODES.keys()))
with col2:
    lingua_dest = st.selectbox("Lingua di Destinazione", list(LANG_CODES.keys()), index=1)

if st.button("Traduci"):
    tokenizer.src_lang = LANG_CODES[lingua_orig]
    encoded_input = tokenizer(testo, return_tensors="pt")
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(LANG_CODES[lingua_dest]))
    risultato = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    st.text_area("Risultato", risultato, disabled=True)