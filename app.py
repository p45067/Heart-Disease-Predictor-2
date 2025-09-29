import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load trained pipeline
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

# ------------------------------
# App Title
# ------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Provide details below to predict the likelihood of heart disease.")

# ------------------------------
# Input Fields (same as dataset columns)
# ------------------------------
age = st.number_input("Age", min_value=20, max_value=100, value=40)
sex = st.selectbox("Sex", options=["M", "F"])
cp = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=["Y", "N"])
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

# ------------------------------
# Prepare input for prediction
# ------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "ChestPainType": cp,
    "RestingBP": trestbps,
    "Cholesterol": chol,
    "FastingBS": fbs,
    "RestingECG": restecg,
    "MaxHR": thalach,
    "ExerciseAngina": exang,
    "Oldpeak": oldpeak,
    "ST_Slope": slope
}])

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Result")
    if prediction == "Yes":
        st.error(f"⚠️ The model predicts **Heart Disease (Yes)** "
                 f"(Confidence: {max(probability):.2f})")
    else:
        st.success(f"✅ The model predicts **No Heart Disease** "
                   f"(Confidence: {max(probability):.2f})")
