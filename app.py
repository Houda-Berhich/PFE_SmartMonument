import streamlit as st
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import subprocess
import unicodedata
import re
import datetime
import os

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="Monuments du Maroc — IA",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #111110;
    --surface:   #1A1A18;
    --surface2:  #222220;
    --border:    #2E2E2A;
    --border2:   #3A3A35;
    --gold:      #C9A84C;
    --gold-dim:  #8A6E2E;
    --red:       #B8293A;
    --text:      #E8E4DC;
    --muted:     #7A7670;
    --faint:     #3D3D38;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── TYPOGRAPHY ── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    font-size: 2.4rem !important;
    letter-spacing: 0.04em !important;
    color: var(--text) !important;
    line-height: 1.15 !important;
    margin-bottom: 0 !important;
}
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    font-size: 1.3rem !important;
    letter-spacing: 0.02em !important;
}
p, label, .stMarkdown p {
    color: var(--muted) !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
}

/* ── HEADER AREA ── */
.app-header {
    padding: 2.5rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-eyebrow {
    font-size: 0.82rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
    margin-bottom: 0.6rem;
    font-family: 'DM Sans', sans-serif;
}
.app-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 300;
    color: var(--text);
    letter-spacing: 0.03em;
    line-height: 1.1;
    margin: 0;
}
.app-title em {
    color: var(--gold);
    font-style: italic;
}
.app-subtitle {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── ONBOARDING GUIDE ── */
.guide-outer {
    border: 1px solid var(--border);
    border-top: 2px solid var(--gold);
    border-radius: 4px;
    background: var(--surface);
    padding: 1.6rem 1.8rem;
    margin-bottom: 2rem;
    animation: fadeSlideIn 0.5s ease both;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.guide-head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
}
.guide-head-label {
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
}
.guide-head-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.45rem;
    font-weight: 400;
    color: var(--text);
    letter-spacing: 0.02em;
}
.guide-steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.guide-step {
    background: var(--surface2);
    padding: 1rem 0.9rem;
    transition: background 0.2s;
}
.guide-step:hover { background: var(--faint); }
.step-num {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold-dim);
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.step-icon {
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
    display: block;
}
.step-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
    letter-spacing: 0.01em;
}
.step-desc {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.5;
    font-weight: 300;
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── GALLERY ── */
[data-testid="stImage"] img {
    border-radius: 3px;
    border: 1px solid var(--border2);
    transition: border-color 0.25s, transform 0.25s;
    filter: brightness(0.92) saturate(0.85);
}
[data-testid="stImage"] img:hover {
    border-color: var(--gold-dim);
    transform: scale(1.015);
    filter: brightness(1) saturate(1);
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 4px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold-dim) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 3px !important;
    font-weight: 400 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.3rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: #0E0E0C !important;
    border-color: var(--gold) !important;
    box-shadow: 0 0 20px rgba(201,168,76,0.15) !important;
}

/* ── DETECTION RESULT ── */
.detection-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 3px;
    padding: 1rem 1.4rem;
    margin: 1.2rem 0;
}
.detection-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 400;
    color: var(--text);
    letter-spacing: 0.02em;
}
.detection-confidence {
    font-size: 0.88rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
}

/* ── INFO CARD ── */
.info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin: 1rem 0;
    animation: fadeSlideIn 0.4s ease both;
}
.info-card-header {
    background: var(--surface2);
    padding: 0.8rem 1.4rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
}
.info-row {
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    padding: 0.9rem 1.4rem;
    border-bottom: 1px solid var(--border);
}
.info-row:last-child { border-bottom: none; }
.info-row-icon { color: var(--gold-dim); font-size: 0.9rem; margin-top: 1px; flex-shrink: 0; }
.info-row-key {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
    min-width: 90px;
    margin-top: 2px;
}
.info-row-val {
    font-size: 1rem;
    color: var(--text);
    font-weight: 300;
    line-height: 1.6;
}
.confidence-bar {
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--gold-dim), var(--gold));
    border-radius: 1px;
    transition: width 1s ease;
}

/* ── CHAT ── */
.chat-shell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 1rem;
    animation: fadeSlideIn 0.4s ease both;
}
.chat-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.4rem;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
}
.chat-topbar-left {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.chat-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.02em;
}
.chat-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.8rem;
    color: #5AAF6A;
    letter-spacing: 0.08em;
}
.chat-status-dot {
    width: 5px; height: 5px;
    background: #5AAF6A;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.chat-body {
    max-height: 360px;
    overflow-y: auto;
    padding: 1.2rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
    scrollbar-width: thin;
    scrollbar-color: var(--border2) transparent;
}
.msg-row { display: flex; gap: 10px; align-items: flex-end; }
.msg-row.user { flex-direction: row-reverse; }
.msg-avatar {
    width: 28px; height: 28px;
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    flex-shrink: 0;
}
.msg-avatar.bot  { background: var(--surface2); border: 1px solid var(--border2); }
.msg-avatar.user { background: var(--gold); color: #0E0E0C; }
.msg-bubble {
    max-width: 75%;
    padding: 0.7rem 1rem;
    font-size: 0.97rem;
    line-height: 1.65;
    font-weight: 300;
}
.msg-bubble.bot {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border2);
    border-radius: 8px 8px 8px 2px;
}
.msg-bubble.user {
    background: var(--gold);
    color: #0E0E0C;
    border-radius: 8px 8px 2px 8px;
    font-weight: 400;
}
.msg-time {
    font-size: 0.72rem;
    color: var(--faint);
    margin-top: 3px;
    padding: 0 3px;
    letter-spacing: 0.05em;
}



/* ── TEXT INPUT ── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.01em !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--gold-dim) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.08) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--faint) !important; }

/* ── PROGRESS ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--gold-dim), var(--gold)) !important;
}

/* ── MISC ── */
.stSpinner > div { border-top-color: var(--gold) !important; }
hr { border-color: var(--border) !important; }
.stCaption { color: var(--muted) !important; font-size: 0.85rem !important; letter-spacing: 0.05em !important; }
.stSuccess, [data-testid="stAlert"] { background: var(--surface) !important; border-color: var(--gold-dim) !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 0.7rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.82rem !important; letter-spacing: 0.1em !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Cormorant Garamond', serif !important; font-size: 1.5rem !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──
CONFIDENCE_THRESHOLD = 0.45
IMAGE_FOLDER = "images"

SUGGESTED_QUESTIONS = [
    "Quand a-t-il été construit ?",
    "Quel est son style architectural ?",
    "Qui l'a fait construire ?",
    "Quelle est son importance historique ?",
    "Peut-on le visiter aujourd'hui ?",
    "Où se trouve-t-il exactement ?",
]

def load_gallery():
    gallery = {}
    if os.path.exists(IMAGE_FOLDER):
        for file in os.listdir(IMAGE_FOLDER):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                name = file.replace(".jpg","").replace(".jpeg","").replace(".png","").replace("_"," ")
                gallery[name] = os.path.join(IMAGE_FOLDER, file)
    return gallery

GALLERY = load_gallery()

# ── MODEL ──
@st.cache_resource
def load_model_and_data():
    model = tf.keras.models.load_model("best_model_phase1_v2.h5")
    with open("monuments_infos.json", "r", encoding="utf-8") as f:
        monuments_info = json.load(f)
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, monuments_info, class_names

model, monuments_info, class_names = load_model_and_data()

# ── HELPERS ──
def normalize_text(text):
    text = text.lower().replace("_"," ").replace("'"," ").replace("'"," ")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()

def find_monument_info(predicted_label, monuments_info):
    predicted_clean = normalize_text(predicted_label)
    for key in monuments_info.keys():
        key_clean = normalize_text(key)
        if predicted_clean in key_clean or key_clean in predicted_clean:
            return monuments_info[key]
    return {}

def ask_llama_about_monument(monument_name, monument_info, question):
    info_text = ", ".join(f"{k}: {v}" for k, v in monument_info.items())
    prompt = f"""Tu es un expert en histoire marocaine.
Monument : {monument_name}
Informations : {info_text}
Question : {question}
Réponds clairement et brièvement."""
    try:
        result = subprocess.run(["ollama","run","llama3",prompt], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Erreur : {e}"

def predict_place(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return class_names[idx], float(np.max(preds))

def now():
    return datetime.datetime.now().strftime("%H:%M")

# ── SESSION STATE ──
for key, default in [
    ("panel", None), ("chat_history", []),
    ("selected_image", None), ("prefill_question", ""),
    ("show_guide", True), ("gallery_page", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-title"><em>SmartMonument</em>— Explorez le patrimoine du Maroc</div>
    <div class="app-subtitle">Une simple photo suffit pour identifier un monument et découvrir son histoire.</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# GUIDE D'UTILISATION
# ═══════════════════════════════════════════════
if st.session_state.show_guide:
    st.markdown("""
    <div class="guide-outer">
        <div class="guide-head">
            <span class="guide-head-label">Guide</span>
            <span class="guide-head-title">Comment utiliser l'application</span>
        </div>
        <div class="guide-steps">
            <div class="guide-step">
                <span class="step-icon">🖼</span>
                <div class="step-num">01 — Sélectionner</div>
                <div class="step-title">Choisir une image</div>
                <div class="step-desc">Uploadez votre photo ou sélectionnez un monument dans la galerie ci-dessous.</div>
            </div>
            <div class="guide-step">
                <span class="step-icon">🔍</span>
                <div class="step-num">02 — Analyser</div>
                <div class="step-title">Détection IA</div>
                <div class="step-desc">Le modèle identifie automatiquement le monument et calcule un score de confiance.</div>
            </div>
            <div class="guide-step">
                <span class="step-icon">📋</span>
                <div class="step-num">03 — Explorer</div>
                <div class="step-title">Consulter la fiche</div>
                <div class="step-desc">Accédez à la localisation, la date de construction et la description complète.</div>
            </div>
            <div class="guide-step">
                <span class="step-icon">💬</span>
                <div class="step-num">04 — Questionner</div>
                <div class="step-title">Dialoguer avec l'IA</div>
                <div class="step-desc">Cliquez sur une question suggérée ou rédigez la vôtre dans le chat.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Masquer le guide"):
        st.session_state.show_guide = False
        st.rerun()
else:
    if st.button("Afficher le guide d'utilisation"):
        st.session_state.show_guide = True
        st.rerun()

# ═══════════════════════════════════════════════
# GALERIE
# ═══════════════════════════════════════════════
st.markdown('<div class="section-label">Proposition d’images de monuments</div>', unsafe_allow_html=True)

IMAGES_PER_PAGE = 3
gallery_items = list(GALLERY.items())
total_pages = max(1, (len(gallery_items) - 1) // IMAGES_PER_PAGE + 1)
start = st.session_state.gallery_page * IMAGES_PER_PAGE
current_items = gallery_items[start:start + IMAGES_PER_PAGE]

if current_items:
    cols = st.columns(len(current_items))
    for i, (name, path) in enumerate(current_items):
        img_g = Image.open(path).convert("RGB").resize((220, 155))
        with cols[i]:
            st.image(img_g)
            st.caption(name.title())
            if st.button("Analyser", key=f"gal_{name}"):
                st.session_state.selected_image = img_g
                st.session_state.panel = None
                st.session_state.chat_history = []
                st.session_state.prefill_question = ""

col_l, col_c, col_r = st.columns([1, 2, 1])
with col_l:
    if st.button("← Précédent") and st.session_state.gallery_page > 0:
        st.session_state.gallery_page -= 1
with col_r:
    if st.button("Suivant →") and st.session_state.gallery_page < total_pages - 1:
        st.session_state.gallery_page += 1
st.caption(f"{st.session_state.gallery_page + 1} / {total_pages}")

# ═══════════════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════════════
st.markdown('<div class="section-label">Analyser une image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Déposez une image (JPG, PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
if uploaded_file:
    st.session_state.selected_image = Image.open(uploaded_file).convert("RGB")
    st.session_state.panel = None
    st.session_state.chat_history = []
    st.session_state.prefill_question = ""

# ═══════════════════════════════════════════════
# ANALYSE & RÉSULTATS
# ═══════════════════════════════════════════════
if st.session_state.selected_image is not None:
    img = st.session_state.selected_image
    st.image(img, use_column_width=True)

    with st.spinner("Identification en cours…"):
        label, confidence = predict_place(img)

    info = find_monument_info(label, monuments_info)

    if confidence < CONFIDENCE_THRESHOLD:
        st.error("Monument non identifié avec une confiance suffisante.")
    else:
        pct = confidence * 100

        # ── Detection banner ──
        st.markdown(f"""
        <div class="detection-banner">
            <div>
                <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;">Monument identifié</div>
                <div class="detection-name">{label}</div>
            </div>
            <div style="text-align:right">
                <div class="detection-confidence">{pct:.1f}% de confiance</div>
                <div class="confidence-bar" style="width:90px;margin-left:auto;margin-top:6px">
                    <div class="confidence-fill" style="width:{pct}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Action buttons ──
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Consulter la fiche complète"):
                st.session_state.panel = "info"
        with c2:
            if st.button("Dialoguer avec l'assistant"):
                st.session_state.panel = "question"
                if not st.session_state.chat_history:
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": f"Bonjour. Je suis RihlaBot, je peux répondre à toutes vos questions concernant {label}. Sélectionnez une question suggérée ou rédigez la vôtre.",
                        "time": now()
                    })

        # ────────────────────────────────────────
        # FICHE INFO
        # ────────────────────────────────────────
        if st.session_state.panel == "info":
            lieu        = info.get("lieu",        "Non disponible")
            date        = info.get("date",        "Non disponible")
            description = info.get("description", "Aucune description disponible")

            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-header">Fiche patrimoniale — {label}</div>
                <div class="info-row">
                    <div class="info-row-icon">🏛</div>
                    <div class="info-row-key">Monument</div>
                    <div class="info-row-val">{label}</div>
                </div>
                <div class="info-row">
                    <div class="info-row-icon">📍</div>
                    <div class="info-row-key">Localisation</div>
                    <div class="info-row-val">{lieu}</div>
                </div>
                <div class="info-row">
                    <div class="info-row-icon">📅</div>
                    <div class="info-row-key">Date</div>
                    <div class="info-row-val">{date}</div>
                </div>
                <div class="info-row">
                    <div class="info-row-icon">📝</div>
                    <div class="info-row-key">Description</div>
                    <div class="info-row-val">{description}</div>
                </div>
                <div class="info-row">
                    <div class="info-row-icon">📊</div>
                    <div class="info-row-key">Confiance IA</div>
                    <div class="info-row-val">
                        {pct:.2f}%
                        <div class="confidence-bar" style="width:180px;margin-top:6px">
                            <div class="confidence-fill" style="width:{pct}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ────────────────────────────────────────
        # CHAT
        # ────────────────────────────────────────
        if st.session_state.panel == "question":

            # Build bubbles HTML
            bubbles_html = ""
            for msg in st.session_state.chat_history:
                role    = msg["role"]
                content = msg["content"]
                t       = msg.get("time","")
                if role == "user":
                    bubbles_html += f"""
                    <div class="msg-row user">
                        <div class="msg-avatar user">Vous</div>
                        <div>
                            <div class="msg-bubble user">{content}</div>
                            <div class="msg-time" style="text-align:right">{t}</div>
                        </div>
                    </div>"""
                else:
                    bubbles_html += f"""
                    <div class="msg-row">
                        <div class="msg-avatar bot">IA</div>
                        <div>
                            <div class="msg-bubble bot">{content}</div>
                            <div class="msg-time">{t}</div>
                        </div>
                    </div>"""

            st.markdown(f"""
            <div class="chat-shell">
                <div class="chat-topbar">
                    <div class="chat-topbar-left">
                        <div class="chat-title">RihlaBot</div>
                    </div>
                    <div class="chat-status">
                        <div class="chat-status-dot"></div>En ligne
                    </div>
                </div>
                <div class="chat-body">{bubbles_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # Functional question buttons
            st.markdown("**Sélectionnez une question :**")
            cols_q = st.columns(2)
            for i, q in enumerate(SUGGESTED_QUESTIONS):
                with cols_q[i % 2]:
                    if st.button(q, key=f"sq_{i}"):
                        st.session_state.chat_history.append({"role":"user","content":q,"time":now()})
                        with st.spinner("Analyse en cours…"):
                            answer = ask_llama_about_monument(label, info, q)
                        st.session_state.chat_history.append({"role":"bot","content":answer,"time":now()})
                        st.rerun()

            st.markdown("---")
            question = st.text_input(
                "Votre question",
                value=st.session_state.prefill_question,
                placeholder="Ex : Qui a ordonné la construction de ce monument ?",
                key="question_input",
                label_visibility="collapsed"
            )
            if st.button("Envoyer") and question.strip():
                st.session_state.prefill_question = ""
                st.session_state.chat_history.append({"role":"user","content":question,"time":now()})
                with st.spinner("Analyse en cours…"):
                    answer = ask_llama_about_monument(label, info, question)
                st.session_state.chat_history.append({"role":"bot","content":answer,"time":now()})
                st.rerun()

