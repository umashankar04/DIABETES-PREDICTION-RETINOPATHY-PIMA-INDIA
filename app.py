%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# Load the trained model
# Make sure the path to your model file is correct
try:
    with open('log_reg_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'log_reg_model.pkl' not found. Please make sure the model file is in the correct directory.")
    st.stop()

# Create the Streamlit app
st.title('Diabetes Prediction App')

st.write("""
Enter the patient's information to predict whether they have diabetes or not.
""")

# Create input fields for the user to enter data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Create a button to make predictions
if st.button('Predict'):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Scale the input data using the same scaler used during training
    # Assuming you have saved the scaler object as well, or retrain it on the full dataset
    # For simplicity, we'll use a new scaler trained on the initial dataframe (df)
    # In a real application, you should save and load the trained scaler
    try:
        scaler = StandardScaler()
        # Fit the scaler on the original data (excluding the Outcome column)
        # Make sure the path to your data file is correct
        scaler.fit(pd.read_csv("/content/drive/MyDrive/Colab Notebooks/diabetes (1).csv").drop('Outcome', axis=1))
        input_data_scaled = scaler.transform(input_data)
    except FileNotFoundError:
         st.error("Original data file '/content/drive/MyDrive/Colab Notebooks/diabetes (1).csv' not found. Cannot scale input data.")
         st.stop()

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.write('Prediction: The person is likely to have diabetes (Yes)')
    else:
        st.write('Prediction: The person is unlikely to have diabetes (No)')
