# Emotion Detector

A simple Streamlit app that detects the predominant emotion in a piece of text using a TF-IDF vectorizer and a Logistic Regression model.

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- NLTK

## Project Structure

```
Emotion-Detector/
├─ app.py
├─ requirements.txt
├─ README.md
├─ models/
│  ├─ emotion_model.pkl
│  └─ tfidf_vectorizer.pkl
├─ styles/
│  └─ style.css
└─ utils/
   └─ cleaner.py
```

## Install and Run Locally

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## How to use
Enter any text into the input area and click **Detect Emotion**. The app will display the top predicted emotion, its confidence, and a short recommendation.

## Model details
- Model: Logistic Regression
- Features: TF-IDF vectorizer
- Reported accuracy: 90%

## Dataset
Emotions Dataset for NLP — available on Kaggle.

## Screenshots
- (Add screenshots here)

## Author
Muhammad Nihal Imran
