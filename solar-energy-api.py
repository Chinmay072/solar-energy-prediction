import pickle

import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained XGBoost model
MODEL_PATH = r"./bst_model.pck"
bst_model = pickle.load(open(MODEL_PATH, "rb"))

# Maximum value to convert the normalized output
MAX = 257622.0

# List of required features
required_features = ['AirTemp', 'Azimuth', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi', 
                     'PrecipitableWater', 'RelativeHumidity', 'SnowDepth', 'SurfacePressure', 
                     'WindDirection10m', 'WindSpeed10m', 'Zenith']

@app.route('/predict', methods=['POST'])
def predict_solar_energy():
    # Get JSON data from the request
    user_input = request.get_json()

    # Ensure all required features are present
    for feature in required_features:
        if feature not in user_input:
            return jsonify({"error": f"Missing required feature: {feature}"}), 400

    # Convert user input to DataFrame
    input_data = pd.DataFrame([user_input])

    # Create a DMatrix for prediction
    dmatrix = xgb.DMatrix(input_data)

    # Predict normalized solar energy generation
    normalized_prediction = bst_model.predict(dmatrix)[0]

    # Convert normalized energy to kWh
    kwh_energy = normalized_prediction * MAX

    # Convert float32 to regular Python float
    normalized_prediction = float(normalized_prediction)
    kwh_energy = float(kwh_energy)

    # Return the prediction as JSON
    response = {
        # "normalized_energy": round(normalized_prediction, 4),
        "Energy": round(kwh_energy, 2)
    }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)
