import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# ---------- LOAD ANIMATION ----------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tno6cg2w.json")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    h1 {
        text-align: center;
        color: #38bdf8;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>🚀 ML Prediction App</h1>", unsafe_allow_html=True)

st_lottie(lottie_ai, height=200)

st.write("Enter the values below to get prediction 👇")

# ---------- LOAD MODEL ----------
model = pickle.load(open("model (4).pkl", "rb"))

# ---------- INPUT FIELDS ----------
# 👉 Change these based on your dataset
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# ---------- PREDICTION ----------
if st.button("Predict 🔍"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]} 🎯")
