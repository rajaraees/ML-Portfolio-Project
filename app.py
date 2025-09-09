# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Page config ---
st.set_page_config(page_title="CKD Predictor", layout="centered")

st.title("🩺 Chronic Kidney Disease (CKD) Risk Predictor")

# --- Load model pipeline ---
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.markdown("Enter patient details and click **Predict** to check CKD risk.")

# --- Input form ---
with st.form("ckd_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["male", "female"])
        blood_pressure = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=120)
        specific_gravity = st.number_input("Specific gravity (urine)", min_value=1.000, max_value=1.040, value=1.015, format="%.3f")
        albumin = st.selectbox("Albumin (0–5)", [0, 1, 2, 3, 4, 5], index=0)
        sugar = st.selectbox("Sugar in urine (0–5)", [0, 1, 2, 3, 4, 5], index=0)
        pus_cell = st.selectbox("Pus cell", ["normal", "abnormal"])
        pus_cell_clumps = st.selectbox("Pus cell clumps", ["absent", "present"])

    with col2:
        bacteria = st.selectbox("Bacteria in urine", ["absent", "present"])
        blood_glucose_random = st.number_input("Random blood glucose (mg/dL)", min_value=0, max_value=1000, value=100)
        blood_urea = st.number_input("Blood urea (mg/dL)", min_value=0, max_value=500, value=30)
        serum_creatinine = st.number_input("Serum creatinine (mg/dL)", min_value=0.0, max_value=50.0, value=1.0, format="%.2f")
        sodium = st.number_input("Sodium (mEq/L)", min_value=0, max_value=200, value=140)
        potassium = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=20.0, value=4.0, format="%.2f")
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=30.0, value=14.0, format="%.2f")

    # Optional fields
    appetite = st.selectbox("Appetite", ["good", "poor"])
    hypertension = st.selectbox("Hypertension", ["no", "yes"])
    diabetes_mellitus = st.selectbox("Diabetes", ["no", "yes"])
    coronary_artery_disease = st.selectbox("Coronary artery disease", ["no", "yes"])
    anemia = st.selectbox("Anemia", ["no", "yes"])
    pedal_edema = st.selectbox("Pedal edema", ["no", "yes"])

    # Submit button
    submitted = st.form_submit_button("Predict")

# --- Build input DataFrame & Predict ---
if submitted:
    input_dict = {
        "age": [age], "gender": [gender], "blood_pressure": [blood_pressure],
        "specific_gravity": [specific_gravity], "albumin": [albumin], "sugar": [sugar],
        "pus_cell": [pus_cell], "pus_cell_clumps": [pus_cell_clumps], "bacteria": [bacteria],
        "blood_glucose_random": [blood_glucose_random], "blood_urea": [blood_urea],
        "serum_creatinine": [serum_creatinine], "sodium": [sodium], "potassium": [potassium],
        "hemoglobin": [hemoglobin], "packed_cell_volume": [np.nan],
        "white_blood_cell_count": [np.nan], "red_blood_cell_count": [np.nan],
        "hypertension": [hypertension], "diabetes_mellitus": [diabetes_mellitus],
        "coronary_artery_disease": [coronary_artery_disease],
        "appetite": [appetite], "anemia": [anemia], "pedal_edema": [pedal_edema]
    }

    input_df = pd.DataFrame(input_dict)

    try:
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        st.metric("CKD Probability", f"{prob:.2%}")
        if pred == 1:
            st.error("Prediction: Positive (CKD)")
        else:
            st.success("Prediction: Negative (No CKD)")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

