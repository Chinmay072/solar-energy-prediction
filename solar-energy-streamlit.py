import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Set page config for dark theme
st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️", layout="wide")

# Custom CSS for dark theme and styling
# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #8E44AD;
        color: white;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #2C2C2C;
        color: white;
    }
    h1, h2, h3 {
        color: #FFD700;
    }
    .stTextInput>label, .stNumberInput>label {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ... rest of the code remains unchanged ...

MAX = 257622.0

def test_model_2(user_input):
    # Load the trained model
    bst = pickle.load(open(r"bst_model.pck", "rb"))

    # Define the correct feature order
    feature_order = ['AirTemp', 'Azimuth', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi', 
                     'PrecipitableWater', 'RelativeHumidity', 'SnowDepth', 'SurfacePressure', 
                     'WindDirection10m', 'WindSpeed10m', 'Zenith']

    # Convert user input to a DataFrame and reorder columns
    input_data = pd.DataFrame([user_input])[feature_order]

    # Create DMatrix
    dmatrix = xgb.DMatrix(input_data)

    # Make prediction (this is already normalized between 0 and 1)
    normalized_prediction = bst.predict(dmatrix)[0]

    return normalized_prediction
    # Load the trained model
    bst = pickle.load(open(r"C:\Life Projects\nasa space app challenge\Solar Energy Prediction\bst_model.pck", "rb"))

    # Convert user input to a DataFrame
    input_data = pd.DataFrame([user_input])

    # Create DMatrix
    dmatrix = xgb.DMatrix(input_data)

    # Make prediction (this is already normalized between 0 and 1)
    normalized_prediction = bst.predict(dmatrix)[0]

    return normalized_prediction

# Streamlit app
st.title('☀️ Solar Energy Predictor')

# User input form
st.header('Enter Weather Parameters')
user_input = {}
user_input['AirTemp'] = st.number_input('Air Temperature (°C)', value=20.6)
user_input['Azimuth'] = st.number_input('Azimuth (degrees)', value=107)
user_input['CloudOpacity'] = st.number_input('Cloud Opacity (0-1)', min_value=0.0, max_value=1.0, value=0.0)
user_input['DewpointTemp'] = st.number_input('Dewpoint Temperature (°C)', value=6.9)
user_input['Dhi'] = st.number_input('Diffuse Horizontal Irradiance (W/m²)', value=59)
user_input['Dni'] = st.number_input('Direct Normal Irradiance (W/m²)', value=805)
user_input['Ebh'] = st.number_input('Direct (Beam) Horizontal Irradiance (W/m²)', value=279)
user_input['Ghi'] = st.number_input('Global Horizontal Irradiance (W/m²)', value=338)
user_input['PrecipitableWater'] = st.number_input('Precipitable Water (mm)', value=8.6)
user_input['RelativeHumidity'] = st.number_input('Relative Humidity (%)', value=41)
user_input['SnowDepth'] = st.number_input('Snow Depth (cm)', value=0)
user_input['SurfacePressure'] = st.number_input('Surface Pressure (hPa)', value=974.9)
user_input['WindDirection10m'] = st.number_input('Wind Direction at 10m (degrees)', value=280)
user_input['WindSpeed10m'] = st.number_input('Wind Speed at 10m (m/s)', value=3.5)
user_input['Zenith'] = st.number_input('Zenith (degrees)', value=70)

if st.button('Predict Solar Energy Generation'):
    try:
        normalized_energy = test_model_2(user_input)
        
        # Convert normalized energy to Watts
        watts_energy = normalized_energy * MAX
        
        st.markdown(f"<h2 style='color: #FFD700;'>Predicted Energy: {watts_energy:.2f} Watts</h2>", unsafe_allow_html=True)
        
        # Optionally, display the normalized prediction
        # st.info(f"Normalized prediction: {normalized_energy:.4f}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# You can add more sections, explanations, or visualizations as needed
