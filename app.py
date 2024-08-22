import streamlit as st
import pandas as pd
import model
streamlit_code = ""
# Streamlit app
st.title('LOS Prediction for Patients')

# User input form
with st.form(key='patient_form'):
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    insurance = st.selectbox('Insurance', options=['Insurance_A', 'Insurance_B', 'Insurance_C'])
    admission_type = st.selectbox('Admission Type', options=['Emergency', 'Elective'])
    treatment_type = st.selectbox('Treatment Type', options=['Treatment_A', 'Treatment_B', 'Treatment_C'])
    dosage = st.number_input('Dosage', min_value=0.0, max_value=100.0, value=60.0)
    #predicted_los = st.number_input('Predicted LOS', min_value=0.0, max_value=20.0, value=7.0)

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Create DataFrame for prediction
        new_patient_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'insurance': [insurance],
            'admission_type': [admission_type],
            'treatment_type': [treatment_type],
            'dosage': [dosage]
            # 'predicted_los': [predicted_los]
        })

        # Predict baseline LOS for new patient
        baseline_los_prediction = model.pipeline_baseline.predict(new_patient_data)
        st.write(f'Predicted Baseline LOS for new patient: {baseline_los_prediction[0]:.2f} days')

        # Predict LOS based on treatment for new patient
        predicted_los_prediction = model.pipeline_predicted.predict(new_patient_data)
        st.write(f'Predicted LOS for new patient with treatment: {predicted_los_prediction[0]:.2f} days')

# with open('streamlit_app.py', 'w') as f:
#     f.write(streamlit_code)