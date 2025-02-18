import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and dataset
pipe = joblib.load('rf_model.pkl')
df = joblib.load('df.pkl')

# Custom CSS for animations
st.markdown("""
<style>
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

@keyframes slideIn {
    0% { transform: translateY(50px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}

.title {
    animation: fadeIn 2s ease-in-out;
}

.sidebar-header {
    animation: slideIn 1.5s ease-in-out;
}

.predict-button {
    animation: fadeIn 3s ease-in-out;
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
}

.predict-button:hover {
    background-color: #45a049;
}

.success-message {
    animation: fadeIn 2s ease-in-out;
    color: green;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<h1 class="title">Wine Quality Prediction</h1>', unsafe_allow_html=True)

st.sidebar.markdown('<h2 class="sidebar-header">Wine Characteristics</h2>', unsafe_allow_html=True)

# Sidebar inputs
fixed_acidity = st.sidebar.number_input('Fixed Acidity', min_value=float(df['fixed_acidity'].min()), max_value=float(df['fixed_acidity'].max()), value=float(df['fixed_acidity'].mean()))
volatile_acidity = st.sidebar.number_input('Volatile Acidity', min_value=float(df['volatile_acidity'].min()), max_value=float(df['volatile_acidity'].max()), value=float(df['volatile_acidity'].mean()))
citric_acid = st.sidebar.number_input('Citric Acid', min_value=float(df['citric_acid'].min()), max_value=float(df['citric_acid'].max()), value=float(df['citric_acid'].mean()))
residual_sugar = st.sidebar.number_input('Residual Sugar', min_value=float(df['residual_sugar'].min()), max_value=float(df['residual_sugar'].max()), value=float(df['residual_sugar'].mean()))
chlorides = st.sidebar.number_input('Chlorides', min_value=float(df['chlorides'].min()), max_value=float(df['chlorides'].max()), value=float(df['chlorides'].mean()))
free_sulfur_dioxide = st.sidebar.number_input('Free Sulfur Dioxide', min_value=float(df['free_sulfur_dioxide'].min()), max_value=float(df['free_sulfur_dioxide'].max()), value=float(df['free_sulfur_dioxide'].mean()))
total_sulfur_dioxide = st.sidebar.number_input('Total Sulfur Dioxide', min_value=float(df['total_sulfur_dioxide'].min()), max_value=float(df['total_sulfur_dioxide'].max()), value=float(df['total_sulfur_dioxide'].mean()))
density = st.sidebar.number_input('Density', min_value=float(df['density'].min()), max_value=float(df['density'].max()), value=float(df['density'].mean()))
pH = st.sidebar.number_input('pH', min_value=float(df['pH'].min()), max_value=float(df['pH'].max()), value=float(df['pH'].mean()))
sulphates = st.sidebar.number_input('Sulphates', min_value=float(df['sulphates'].min()), max_value=float(df['sulphates'].max()), value=float(df['sulphates'].mean()))
alcohol = st.sidebar.number_input('Alcohol', min_value=float(df['alcohol'].min()), max_value=float(df['alcohol'].max()), value=float(df['alcohol'].mean()))

if st.sidebar.button('Predict Wine Quality', key='predict-button'):
    try:
        # Create a DataFrame for the input
        query = pd.DataFrame({
            'fixed_acidity': [fixed_acidity],
            'volatile_acidity': [volatile_acidity],
            'citric_acid': [citric_acid],
            'residual_sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free_sulfur_dioxide': [free_sulfur_dioxide],
            'total_sulfur_dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol]
        })

        # Predict wine quality
        predicted_quality = pipe.predict(query)[0]
        st.markdown(f'<p class="success-message">Predicted Wine Quality: {predicted_quality}</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")