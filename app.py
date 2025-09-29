import streamlit as st
import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load('random_forest_model.joblib')

st.title('Heart Disease Prediction App')

st.write("""
This app predicts the likelihood of heart disease based on your inputs.
""")

# Get user input
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', ['F', 'M'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.slider('Cholesterol', 0, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 202, 150)
exercise_angina = st.selectbox('Exercise Induced Angina', ['N', 'Y'])
oldpeak = st.slider('Oldpeak (ST depression induced by exercise relative to rest)', 0.0, 6.2, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])


# Create a dataframe from user input
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

# Preprocess the input data (one-hot encoding - consistent with training)
input_data_encoded = pd.get_dummies(input_data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# Ensure all columns from training data are present in input data and in the same order
# This is crucial for the model prediction
for col in X_train.columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

input_data_encoded = input_data_encoded[X_train.columns]


# Predict and display the result
if st.button('Predict'):
    prediction = model.predict(input_data_encoded)
    prediction_proba = model.predict_proba(input_data_encoded)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write('Based on the provided information, there is a high likelihood of heart disease.')
    else:
        st.write('Based on the provided information, there is a low likelihood of heart disease.')

    st.subheader('Prediction Probability')
    st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
