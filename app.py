import streamlit as st
import joblib
import numpy as np

# Load ML model and encoders
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
smoking_encoder = joblib.load('smoking_encoder.pkl')

st.set_page_config(page_title='Diabetes Prediction', page_icon='🩺')

st.title('🩺 Diabetes Prediction System')
st.write('Enter your health details below to predict diabetes risk.')

# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 1, 100, 25)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
smoking_history = st.selectbox(
    'Smoking History',
    ['never', 'current', 'former', 'No Info', 'ever', 'not current']
)
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=22.0)
hba1c = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, value=5.5)
blood_glucose = st.number_input('Blood Glucose Level', min_value=50, max_value=300, value=100)

if st.button('🔍 Predict Diabetes'):
    gender_value = gender_encoder.transform([gender])[0]
    smoking_value = smoking_encoder.transform([smoking_history])[0]

    patient = np.array([[gender_value,
                         age,
                         1 if hypertension == 'Yes' else 0,
                         1 if heart_disease == 'Yes' else 0,
                         smoking_value,
                         bmi,
                         hba1c,
                         blood_glucose]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1] * 100

    st.subheader(f'Prediction Confidence: {probability:.2f}%')

    if prediction == 1:
        st.error('🔴 Positive Diabetes Prediction')
        st.markdown('### Precautions')
        st.write('• Eat a balanced low-sugar diet.')
        st.write('• Exercise at least 30 minutes daily.')
        st.write('• Drink plenty of water.')
        st.write('• Monitor blood sugar regularly.')
        st.write('• Maintain a healthy body weight.')
        st.write('• Consult a doctor for confirmation.')
    else:
        st.success('🟢 Negative Diabetes Prediction')
        st.markdown('### Healthy Lifestyle Tips')
        st.write('• Maintain a healthy balanced diet.')
        st.write('• Exercise regularly.')
        st.write('• Sleep for 7–8 hours daily.')
        st.write('• Reduce sugar intake.')
        st.write('• Get regular health checkups.')