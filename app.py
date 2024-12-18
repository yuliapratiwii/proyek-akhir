import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('./model/rdf_model.joblib')
scaler = joblib.load('./model/scaler.pkl')

# Function to make predictions
# Fungsi untuk membuat prediksi
def predict_status(inputs):
    # Ubah inputs menjadi array numpy dan reshape
    input_array = np.array(inputs).reshape(1, -1)
    
    # Debugging: Cek bentuk input sebelum scaling
    print(f"Bentuk array input sebelum scaling: {input_array.shape}")
    
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    return prediction

# Streamlit UI
st.title('Student Dropout Prediction')

# Input fields for user to input data
curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Semester Approved', min_value=0, max_value=30, value=15)
curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Semester Grade', min_value=0, max_value=20, value=15)
curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Semester Approved', min_value=0, max_value=30, value=15)
curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Semester Grade', min_value=0, max_value=20, value=15)
tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
scholarship_holder = st.selectbox('Scholarship Holder', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Semester Enrolled', min_value=0, max_value=30, value=20)
curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Semester Enrolled', min_value=0, max_value=30, value=20)
admission_grade = st.slider('Admission Grade', min_value=0.0, max_value=200.0, value=5.0, step=0.1)
displaced = st.selectbox('Displaced', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Map the inputs to the format expected by the model
input_data = [
    curricular_units_2nd_sem_approved,
    curricular_units_2nd_sem_grade,
    curricular_units_1st_sem_approved,
    curricular_units_1st_sem_grade,
    tuition_fees_up_to_date,
    scholarship_holder,
    curricular_units_2nd_sem_enrolled,
    curricular_units_1st_sem_enrolled,
    admission_grade,
    displaced
]

# Button for prediction
if st.button('Predict'):
    prediction = predict_status(input_data)

    # The prediction will be a 2D array where each column corresponds to one of the classes
    status_dict = {
        0: 'Dropout',
        1: 'Enrolled',
        2: 'Graduate'
    }
    # Find the index of the maximum predicted value
    predicted_status_index = np.argmax(prediction, axis=1)[0]
    predicted_status = status_dict[predicted_status_index]

    st.write(f"The model predicts that the student is likely to be: **{predicted_status}**")
