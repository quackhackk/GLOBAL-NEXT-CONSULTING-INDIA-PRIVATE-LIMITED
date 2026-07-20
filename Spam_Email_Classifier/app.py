import streamlit as st
import joblib
from pathlib import Path
import traceback

BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "random_forest_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

    st.title("Spam Email Classifier")

    email = st.text_area("Enter Email")

    if st.button("Predict"):
        x = vectorizer.transform([email])
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0]

        if pred == 1:
            st.error(f"Spam ({prob[1]*100:.2f}%)")
        else:
            st.success(f"Ham ({prob[0]*100:.2f}%)")

except Exception:
    st.error("An error occurred")
    st.code(traceback.format_exc())