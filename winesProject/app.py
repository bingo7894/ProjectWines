import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #722F37;
        font-size: 3rem !important;
        padding-bottom: 2rem;
    }
    .stSidebar {
        background-color: #f5f5f5;
        padding: 2rem;
    }
    .stButton button {
        background-color: #722F37;
        color: white;
        width: 100%;
    }
    .stSuccess {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and dataset
@st.cache_resource
def load_data():
    pipe = joblib.load('voting_classifier.pkl')
    df = joblib.load('df.pkl')
    return pipe, df

pipe, df = load_data()

# Main title with decorative elements
st.markdown("# 🍷 Wine Quality Prediction 🍷")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Input Parameters")
    
    # Create input fields with more information
    input_params = {
        'fixed_acidity': ('Fixed Acidity (g/dm³)', 'Acids that do not evaporate readily'),
        'volatile_acidity': ('Volatile Acidity (g/dm³)', 'Steam-distillable acids'),
        'citric_acid': ('Citric Acid (g/dm³)', 'Adds freshness and flavor'),
        'residual_sugar': ('Residual Sugar (g/dm³)', 'Amount of sugar remaining after fermentation'),
        'chlorides': ('Chlorides (g/dm³)', 'Amount of salt'),
        'free_sulfur_dioxide': ('Free Sulfur Dioxide (mg/dm³)', 'Prevents microbial growth and oxidation'),
        'total_sulfur_dioxide': ('Total Sulfur Dioxide (mg/dm³)', 'Total amount of SO2'),
        'density': ('Density (g/cm³)', 'Mass per unit volume'),
        'pH': ('pH', 'Acidity level (0-14 scale)'),
        'sulphates': ('Sulphates (g/dm³)', 'Wine additive'),
        'alcohol': ('Alcohol (%)', 'Percent alcohol content')
    }

    inputs = {}
    for param, (label, help_text) in input_params.items():
        inputs[param] = st.number_input(
            label,
            min_value=float(df[param].min()),
            max_value=float(df[param].max()),
            value=float(df[param].mean()),
            help=help_text
        )

    predict_button = st.button('Predict Wine Quality 🔍', use_container_width=True)

with col2:
    st.markdown("### Results and Analysis")
    
    if predict_button:
        try:
            # Create prediction DataFrame
            query = pd.DataFrame({param: [value] for param, value in inputs.items()})
            
            # Make prediction
            predicted_quality = pipe.predict(query)[0]
            
            # Display prediction with styling
            st.markdown(f"""
                <div style='background-color: #f0f8f0; padding: 2rem; border-radius: 0.5rem;'>
                    <h2 style='color: #722F37; text-align: center;'>Predicted Wine Quality</h2>
                    <h1 style='text-align: center; font-size: 4rem;'>{predicted_quality}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Quality interpretation
            quality_map = {
                range(0, 3): "Poor",
                range(4, 5): "Average",
                range(6, 7): "Good",
                range(8): "Excellent"
            }
            
            quality_text = next(desc for quality_range, desc in quality_map.items() 
                              if predicted_quality in quality_range)
            
            st.markdown(f"""
                <div style='text-align: center; margin-top: 1rem;'>
                    <h3>Quality Category: {quality_text}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Display wine image
            st.image('1f377.gif', use_container_width=True)
            
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    else:
        st.info("👈 Adjust the parameters on the left and click 'Predict' to get the wine quality prediction")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Made with ❤️ for wine enthusiasts</p>
        <p>Data source: UCI Machine Learning Repository - Wine Quality Dataset</p>
    </div>
""", unsafe_allow_html=True)