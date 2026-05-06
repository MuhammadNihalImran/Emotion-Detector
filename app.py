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
st.markdown('<h1>🎭 Emotion Detector</h1>', unsafe_allow_html=True)
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
            proba = model.predict_proba(vectorized)[0]
            classes = model.classes_
            
            # Combine classes with their probabilities
            emotion_probs = [{'emotion': cls, 'probability': prob * 100} for cls, prob in zip(classes, proba)]
            
            # 1. Filter out emotions with probability <= 40%
            filtered_emotions = [ep for ep in emotion_probs if ep['probability'] > 40.0]
            
            # 2. Sort remaining emotions from highest to lowest probability
            sorted_emotions = sorted(filtered_emotions, key=lambda x: x['probability'], reverse=True)

        # Result Box
        if not sorted_emotions:
            st.markdown(f"""
            <div style="background:#E3F2FD;border:1px solid #1565C0;border-radius:12px;padding:24px;text-align:center;margin-top:16px;">
                <h1 style="color:#000000; font-size:1.5rem; font-weight:700; margin:0;">No strong emotion detected</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("### Detected Emotions:")
            
            for ep in sorted_emotions:
                emo = ep['emotion']
                conf = round(ep['probability'], 1)
                st.markdown(f"""
                <div style="background:{COLOR_MAP[emo]};border:1px solid {BORDER_MAP[emo]};border-radius:12px;padding:16px;margin-top:12px;">
                    <h2 style="color:#000000; font-size:1.4rem; margin:0; font-weight:600;">{EMOJI_MAP[emo]} {emo.capitalize()}: {conf}%</h2>
                </div>
                """, unsafe_allow_html=True)

            # Keep recommendation for the top emotion
            top_emotion = sorted_emotions[0]['emotion']
            st.markdown("### Recommendation")
            recommendation_text = RECOMMENDATION_MAP.get(top_emotion, "Take a moment and reflect on how you feel.")
            
            st.markdown(f"""
            <div style="background:#1E2330;border:1px solid #00BFFF;border-radius:12px;padding:20px;color:#FFFFFF;font-size:1rem;line-height:1.6;">
                {recommendation_text}
            </div>
            """, unsafe_allow_html=True)