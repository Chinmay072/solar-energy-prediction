import itertools
import matplotlib
import xgboost as xgb
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import time
from dateutil.parser import parse
import pytz

# https://xgboost.readthedocs.io/en/latest/python/python_intro.html

MAX = 257622.0

def test_model_2(user_input):
    # Load the trained model
    bst = pickle.load(open(r"C:\Life Projects\nasa space app challenge\Solar Energy Prediction\bst_model.pck", "rb"))

    # Convert user input to a DataFrame
    input_data = pd.DataFrame([user_input])

    # Ensure all required features are present
    required_features = ['AirTemp', 'Azimuth', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi', 
                         'PrecipitableWater', 'RelativeHumidity', 'SnowDepth', 'SurfacePressure', 
                         'WindDirection10m', 'WindSpeed10m', 'Zenith']
    
    for feature in required_features:
        if feature not in input_data.columns:
            raise ValueError(f"Missing required feature: {feature}")

    # Create DMatrix
    dmatrix = xgb.DMatrix(input_data)

    # Make prediction (this is already normalized between 0 and 1)
    normalized_prediction = bst.predict(dmatrix)[0]

    return normalized_prediction

if __name__ == "__main__":
    # Uncomment these lines if you need to preprocess data or train the model
    # preprocess_data()
    # train_model()
    # Example user input
    user_input = {
        'AirTemp': 20.6,
        'Azimuth': 107,
        'CloudOpacity': 0,
        'DewpointTemp': 6.9,
        'Dhi': 59,
        'Dni': 805,
        'Ebh': 279,
        'Ghi': 338,
        'PrecipitableWater': 8.6,
        'RelativeHumidity': 41,
        'SnowDepth': 0,
        'SurfacePressure': 974.9,
        'WindDirection10m': 280,
        'WindSpeed10m': 3.5,
        'Zenith': 70
    }

    try:
        normalized_energy = test_model_2(user_input)
        
        # Convert normalized energy to kWh
        # Assuming MAX is in Watts and represents 15-minute intervals
        kwh_energy = (normalized_energy * MAX) # / 1000 / 4  # Convert W to kW and 15 min to 1 hour
        
        print(f"Predicted solar energy generation (normalized): {normalized_energy:.4f}")
        print(f"Predicted solar energy generation: {kwh_energy:.2f}")
    except ValueError as e:
        print(f"Error: {e}")