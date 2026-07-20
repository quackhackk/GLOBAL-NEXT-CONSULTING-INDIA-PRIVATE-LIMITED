import streamlit as st
import joblib
from pathlib import Path
import traceback

BASE_DIR = Path(__file__).resolve().parent

st.write("App directory:", BASE_DIR)
st.write("Files:", [p.name for p in BASE_DIR.iterdir()])

try:
    model = joblib.load(BASE_DIR / "random_forest_model.pkl")
    st.success("Model loaded successfully!")

    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    st.success("Vectorizer loaded successfully!")

except Exception as e:
    st.error(type(e).__name__)
    st.code(traceback.format_exc())