import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import json

# 1. Caricamento del modello in cache per risparmiare memoria
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Glossario codici lingue supportate dal bot
LANG_CODES = {
    "Italiano": "it", "Inglese": "en", "Spagnolo": "es", "Francese": "fr", 
    "Tedesco": "de", "Giapponese": "ja", "Russo": "ru", "Vietnamita": "vi", 
    "Arabo": "ar", "Ceco": "cs", "Indonesiano": "id", "Portoghese (Brasile)": "pt", 
    "Polacco": "pl"
}

# 2. Funzione centrale di traduzione neurale
def esegui_traduzione(testo, iso_orig, iso_dest):
    if not testo.strip() or iso_orig == iso_dest:
        return testo
    try:
        tokenizer.src_lang = iso_orig
        encoded_input = tokenizer(testo, return_tensors="pt")
        generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(iso_dest))
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        return f"[Errore Modello]: {str(e)}"

# 3. INTERCETTAZIONE DELLA CHIAMATA API DEL BOT (In cima per evitare l'HTML)
query_params = st.query_params
if "text" in query_params and "from" in query_params and "to" in query_params:
    testo_api = query_params["text"]
    da_iso = query_params["from"]
    a_iso = query_params["to"]
    
    traduzione_api = esegui_traduzione(testo_api, da_iso, a_iso)
    
    # Restituisce SOLO il JSON puro al bot Discord
    st.write(json.dumps({"translated_text": traduzione_api}))
    st.stop()

# 4. INTERFACCIA WEB STANDARD (Visualizzata solo se apri il link dal browser)
st.title("🌍 Il Mio Traduttore Indipendente")
testo_input = st.text_area("Testo da tradurre", placeholder="Scrivi qui...")
col1, col2 = st.columns(2)
with col1:
    lingua_orig = st.selectbox("Lingua di Partenza", list(LANG_CODES.keys()))
with col2:
    lingua_dest = st.selectbox("Lingua di Destinazione", list(LANG_CODES.keys()), index=1)

if st.button("Traduci"):
    risultato = esegui_traduzione(testo_input, LANG_CODES[lingua_orig], LANG_CODES[lingua_dest])
    st.text_area("Risultato", risultato, disabled=True)