
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_LABEL = "Model_2_Clinical_Lab"
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title=MODEL_LABEL, layout="centered")
st.title(MODEL_LABEL.replace("_", " "))
st.caption("Exploratory clinical stratification tool for single viral detection in ARTI patients. Internal validation only.")

model = joblib.load(BASE_DIR / f"{MODEL_LABEL}_selected_model.joblib")
with open(BASE_DIR / f"{MODEL_LABEL}_selected_features.json", "r", encoding="utf-8") as f:
    selected_features = json.load(f)
with open(BASE_DIR / f"{MODEL_LABEL}_feature_schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

st.warning("This tool is exploratory and should not be used as a definitive diagnostic tool.")
user_input = {}

with st.form("prediction_form"):
    for feat in selected_features:
        info = schema.get(feat, {"type": "numeric"})
        if info.get("type") == "numeric":
            user_input[feat] = st.number_input(feat, value=float(info.get("median", 0.0)), format="%.3f")
        else:
            vals = info.get("values", [""])
            user_input[feat] = st.selectbox(feat, vals)
    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([user_input], columns=selected_features)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_new)[:, 1][0])
    else:
        score = float(model.decision_function(X_new)[0])
        prob = 1 / (1 + np.exp(-score))
    st.subheader("Predicted probability")
    st.metric("Probability of single viral detection", f"{prob:.3f}")
    st.caption("Outcome=1: single viral detection; Outcome=0: bacterial or mixed pathogen detection.")
