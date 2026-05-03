import streamlit as st
import joblib
from utils.cleaner import clean_text

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────
css = open("styles/style.css", "r", encoding="utf-8").read()
# Ensure CSS is injected before any other Streamlit components render
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Also explicitly hide Streamlit default menu/header/footer for a clean UI
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/emotion_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        return model, tfidf
    except Exception as e:
        st.error(f"Model files not found or failed to load: {e}")
        return None, None

model, tfidf = load_model()
if model is None or tfidf is None:
    st.stop()

# ── Config ────────────────────────────────────────────
EMOJI_MAP = {
    'joy': '😊', 'sadness': '😢', 'anger': '😡',
    'fear': '😨', 'love': '❤️', 'surprise': '😲'
}
COLOR_MAP = {
    'joy': '#FFF8E1', 'sadness': '#E3F2FD', 'anger': '#FFEBEE',
    'fear': '#F3E5F5', 'love': '#FCE4EC', 'surprise': '#E0F7FA'
}
BORDER_MAP = {
    'joy': '#F9A825', 'sadness': '#1565C0', 'anger': '#C62828',
    'fear': '#6A1B9A', 'love': '#AD1457', 'surprise': '#00838F'
}
RECOMMENDATION_MAP = {
    'joy': 'Maintain this energy. Share it with people around you.',
    'sadness': 'Reach out to someone you trust. A short walk or calm music can help shift your mood.',
    'anger': 'Step away briefly. Practice deep breathing before responding to anything.',
    'fear': 'Ground yourself. Focus on your breathing and speak to someone you feel safe with.',
    'love': 'Express it. Let the people who matter know how you feel.',
    'surprise': 'Take a moment to process. Journaling can help bring clarity.'
}

# ── UI ────────────────────────────────────────────────
st.title("🎭 Emotion Detector")
st.caption("Enter any text and the model will detect the underlying emotion.")
st.divider()

user_input = st.text_area(
    "Enter your text:",
    placeholder="e.g. I'm so excited about tomorrow!",
    height=120,
    label_visibility="collapsed"
)

if st.button("Detect Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            emotion = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence = round(max(proba) * 100, 1)

        # Result Box
        st.markdown(f"""
        <div class="result-box" style="background:{COLOR_MAP[emotion]};border-color:{BORDER_MAP[emotion]};">
            <p class="emotion-label">{EMOJI_MAP[emotion]} {emotion.capitalize()}</p>
            <p class="confidence">Confidence: {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Recommendation")
        recommendation_text = RECOMMENDATION_MAP.get(emotion, "Take a moment and reflect on how you feel.")
        if confidence < 70:
            recommendation_text = "Confidence is low. Try rephrasing your input for a better result."

        st.markdown(f"""
        <div class="recommendation-box">
            {recommendation_text}
        </div>
        """, unsafe_allow_html=True)