import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 1. Caricamento del modello in cache per risparmiare memoria
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

LANG_CODES = {"Italiano": "it", "Inglese": "en", "Spagnolo": "es", "Francese": "fr", "Tedesco": "de", "Giapponese": "ja", "Russo": "ru"}

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

# 3. INTERFACCIA WEB (Streamlit Standard)
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

# 4. PORTA SUL RETRO PER IL BOT DI DISCORD (Query Parameters)
# Permette al bot di fare una richiesta web silenziosa passando i dati nell'URL
query_params = st.query_params
if "text" in query_params and "from" in query_params and "to" in query_params:
    testo_api = query_params["text"]
    da_iso = query_params["from"]
    a_iso = query_params["to"]
    
    traduzione_api = esegui_traduzione(testo_api, da_iso, a_iso)
    # Mostra la risposta JSON pulita sullo schermo se chiamata via codice
    st.json({"translated_text": list(traduzione_api) if isinstance(traduzione_api, list) else traduzione_api})