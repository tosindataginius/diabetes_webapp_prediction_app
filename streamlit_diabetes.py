# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 02:22:02 2025

@author: dataginius
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Page config ---
st.set_page_config(page_title="Diabetes Clinical Predictor", layout="centered", initial_sidebar_state="expanded")

# --- Styling ---
st.markdown(
    """
    <style>
    .reportview-container {background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);} 
    .stButton>button {border-radius: 10px;}
    .big-font {font-size:20px; font-weight:600}
    .muted {color: #6c757d; font-size:13px}
    .card {background: #ffffff; padding: 18px; border-radius:12px; box-shadow: 0 4px 14px rgba(20,30,60,0.08);}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helpers ---

#MODEL_FILENAME = joblib.load(r"C:Users\\USER\\Downloads\\Machine\\diabetes_prediction_app\\diabetes_prediction_model.joblib")
MODEL_FILENAME = r"C:\\Users\\USER\\Downloads\\Machine\\diabetes_prediction_app\\diabetes_prediction_model.joblib"


DEFAULT_FEATURES = [
    'Age',
    'Alcohol Consumption per Week',
    'Physical Activity Minutes per Week',
    'Sleep Hours per Day',
    'Screen Time Hours per Day',
    'Family History Diabetes',
    'Hypertension History',
    'Cardiovascular History',
    'BMI',
    'Systolic bp',
    'Diastolic bp',
    'Cholesterol Total',
    'Glucose Fasting',
    'Glucose Postprandial',
    'Insulin Level',
    'Smoking Status'
]

FEATURE_EXPLANATIONS = {
    'Age': 'Age of the patient in years.',
    'Alcohol Consumption per Week': 'Number of alcoholic drinks consumed per week — excessive intake raises metabolic risks.',
    'Physical Activity Minutes per Week': 'Weekly minutes of physical activity. Higher values reduce diabetes risk.',
    'Sleep Hours per Day': 'Average sleep duration per day. Both too little and too much sleep affect metabolism.',
    'Screen Time Hours per Day': 'Average daily screen exposure. Prolonged sedentary screen time increases diabetes risk.',
    'Family History Diabetes': '1 if patient has a family history of diabetes; 0 otherwise.',
    'Hypertension History': '1 if patient has a history of hypertension (high blood pressure); 0 otherwise.',
    'Cardiovascular History': '1 if patient has cardiovascular disease history; 0 otherwise.',
    'BMI': 'Body Mass Index (weight/height²). BMI > 25 indicates overweight; > 30 obesity.',
    'Systolic bp': 'Systolic blood pressure (mmHg).',
    'Diastolic bp': 'Diastolic blood pressure (mmHg).',
    'Cholesterol Total': 'Total serum cholesterol level (mg/dL).',
    'Glucose Fasting': 'Fasting plasma glucose concentration (mg/dL).',
    'Glucose Postprandial': '2-hour postprandial glucose level (mg/dL).',
    'Insulin Level': 'Fasting insulin concentration (μU/mL).',
    'Smoking Status': 'Smoking status (1 = smoker, 0 = non-smoker).'
}

@st.cache_data
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"Model file not found at {path}. Place your joblib model in the app folder named '{MODEL_FILENAME}'."
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

# --- Sidebar navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5935/5935562.png", width=120)
    st.title("Diabetes Predictor")
    page = st.radio("Go to", ["Predict", "Feature Info", "Model Info", "About"], index=0)
    st.markdown("---")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    st.caption("Built for clinicians — quick inputs, clear outputs.")

# --- Load model ---
model, load_err = load_model(MODEL_FILENAME)

# --- Pages ---
if page == "Predict":
    st.markdown("## Clinical Prediction — Diabetes Risk🏥")
    st.markdown("<div class='muted'>Enter patient data below.</div>", unsafe_allow_html=True)

    with st.form(key='input_form'):
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input('Age (years)', min_value=0, max_value=120, value=48)
            alcohol = st.number_input('Alcohol consumption per week', min_value=0, max_value=50, value=0)
            activity = st.number_input('Physical activity (min/week)', min_value=0, max_value=1000, value=143)
            sleep = st.number_input('Sleep hours per day', min_value=0.0, max_value=24.0, value=6.5)
            screen = st.number_input('Screen time (hours/day)', min_value=0.0, max_value=24.0, value=8.7)
        with cols[1]:
            fam_hist = st.selectbox('Family history of diabetes', [0, 1])
            hyper = st.selectbox('Hypertension history', [0, 1])
            cardio = st.selectbox('Cardiovascular history', [0, 1])
            bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=23.1)
            sys_bp = st.number_input('Systolic BP (mmHg)', min_value=50, max_value=250, value=129)
        with cols[2]:
            dia_bp = st.number_input('Diastolic BP (mmHg)', min_value=30, max_value=150, value=76)
            chol = st.number_input('Total cholesterol (mg/dL)', min_value=50, max_value=400, value=116)
            glu_fast = st.number_input('Fasting glucose (mg/dL)', min_value=50, max_value=300, value=93)
            glu_post = st.number_input('Postprandial glucose (mg/dL)', min_value=50, max_value=400, value=150)
            insulin = st.number_input('Insulin level (μU/mL)', min_value=0.0, max_value=1000.0, value=2.0)
            smoking = st.selectbox('Smoking status', [0, 1])

        submit = st.form_submit_button("Predict risk")

    input_dict = {
        'age': age,
        'alcohol_consumption_per_week': alcohol,
        'physical_activity_minutes_per_week': activity,
        'sleep_hours_per_day': sleep,
        'screen_time_hours_per_day': screen,
        'family_history_diabetes': fam_hist,
        'hypertension_history': hyper,
        'cardiovascular_history': cardio,
        'bmi': bmi,
        'systolic_bp': sys_bp,
        'diastolic_bp': dia_bp,
        'cholesterol_total': chol,
        'glucose_fasting': glu_fast,
        'glucose_postprandial': glu_post,
        'insulin_level': insulin,
        'Smoking_Status_Encoded': smoking
    }

    input_df = pd.DataFrame([input_dict])

    if submit:
        if model is None:
            st.error(load_err or "No model loaded.")
        else:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)[0][1]
                else:
                    pred = model.predict(input_df)[0]
                    proba = float(pred)

                risk_pct = proba * 100
                st.metric("Predicted risk (%)", f"{risk_pct:.1f}%")
                label = "Diabetes likely" if proba >= threshold else "Diabetes unlikely"
                st.write(f"**Interpretation:** {label}")
                st.progress(int(proba * 100))
                st.table(input_df.T)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Feature Info":
    st.markdown("# Feature explanations (for clinicians)")
    for f in DEFAULT_FEATURES:
        with st.expander(f):
            st.markdown(f"**Definition:** {FEATURE_EXPLANATIONS.get(f, 'Description not available.')} ")
            st.markdown("---")

elif page == "Model Info":
    st.markdown("# Model and deployment information")
    if model is None:
        st.warning(load_err or "No model loaded.")
    else:
        st.success("Model loaded successfully.")
        st.write(f"Model type: {type(model)}")

else:
    st.markdown("# About this app")
    st.markdown("Clinician-focused diabetes risk predictor based on lifestyle, vitals, and lab data.")

st.markdown("---")
st.caption("This tool aids but does not replace clinical judgment.")
