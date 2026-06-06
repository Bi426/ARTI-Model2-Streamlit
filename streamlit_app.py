import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_LABEL = "Model_2_Clinical_Lab"
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Model 2 Clinical-Lab Risk Calculator",
    layout="centered"
)

st.markdown("""
<style>
.main .block-container {
    max-width: 850px;
    padding-top: 2rem;
}
h1 {
    font-size: 2.4rem !important;
}
.input-card {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    border: 1px solid #e6e6e6;
    background-color: #fafafa;
    margin-bottom: 1rem;
}
.result-box {
    padding: 1.2rem;
    border-radius: 16px;
    background-color: #f3f7ff;
    border: 1px solid #d6e4ff;
}
</style>
""", unsafe_allow_html=True)

st.title("Model 2 Clinical-Lab Risk Calculator")
st.caption(
    "Exploratory clinical stratification tool for estimating the probability of single viral detection in ARTI patients."
)

st.warning(
    "This tool is intended for exploratory risk stratification only and should not be used as a definitive diagnostic tool."
)

model = joblib.load(BASE_DIR / f"{MODEL_LABEL}_selected_model.joblib")

with open(BASE_DIR / f"{MODEL_LABEL}_selected_features.json", "r", encoding="utf-8") as f:
    selected_features = json.load(f)

with open(BASE_DIR / f"{MODEL_LABEL}_feature_schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

def pretty_name(x):
    return x.replace(".", " ").replace("_", " ")

binary_features = [
    "Type.2.diabetes.mellitus",
    "Cough",
    "Hypertension",
    "Expectoration",
    "Fever",
    "Sore.throat",
    "Dyspnea"
]

user_input = {}

st.markdown("### Patient information")

with st.form("prediction_form"):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    cols = st.columns(2)

    for i, feat in enumerate(selected_features):
        info = schema.get(feat, {"type": "numeric"})
        col = cols[i % 2]

        with col:
            label = pretty_name(feat)

            if feat in binary_features:
                val = st.selectbox(
                    label,
                    options=["No", "Yes"],
                    index=0,
                    key=feat
                )
                user_input[feat] = 1 if val == "Yes" else 0

            elif "age" in feat.lower():
                user_input[feat] = st.number_input(
                    label,
                    value=int(round(float(info.get("median", 0)))),
                    step=1,
                    format="%d",
                    key=feat
                )

            else:
                user_input[feat] = st.number_input(
                    label,
                    value=round(float(info.get("median", 0.0)), 2),
                    step=0.01,
                    format="%.2f",
                    key=feat
                )

    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Calculate risk")

if submitted:
    X_new = pd.DataFrame([user_input], columns=selected_features)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_new)[:, 1][0])
    else:
        score = float(model.decision_function(X_new)[0])
        prob = 1 / (1 + np.exp(-score))

    if prob < 0.30:
        risk_level = "Low risk"
        interpretation = "The predicted probability of single viral detection is low."
    elif prob < 0.70:
        risk_level = "Intermediate risk"
        interpretation = "The predicted probability of single viral detection is intermediate. Further clinical assessment or pathogen testing may be considered."
    else:
        risk_level = "High risk"
        interpretation = "The predicted probability of single viral detection is high. Viral testing and cautious antibiotic decision-making may be considered."

    st.markdown("### Prediction result")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("Predicted probability of single viral detection", f"{prob * 100:.1f}%")
    st.write(f"**Risk level:** {risk_level}")
    st.write(interpretation)
    st.caption("Outcome = 1: single viral detection; Outcome = 0: bacterial or mixed pathogen detection.")
    st.markdown('</div>', unsafe_allow_html=True)