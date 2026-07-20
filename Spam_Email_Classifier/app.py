import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.write("App directory:", BASE_DIR)
st.write("Files:", [p.name for p in BASE_DIR.iterdir()])

model = joblib.load(BASE_DIR / "random_forest_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")