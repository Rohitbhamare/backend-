from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from database import db_instance  # Ensure you have a working database.py

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for your frontend and localhost (if testing locally)
CORS(app, resources={r"/*": {"origins": [
    "https://enchanting-biscochitos-2ee8a8.netlify.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

# Global variables
model = None
preprocessor = None
dataset_info = None

def load_model_and_preprocessor():
    global model, preprocessor, dataset_info
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('dataset_info.pkl', 'rb') as f:
            dataset_info = pickle.load(f)
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")

def preprocess_input(data):
    try:
        df = pd.DataFrame([data])
        current_year = datetime.now().year
        df['car_age'] = current_year - df['year']

        required_features = ['make', 'model', 'year', 'fuel', 'kms_driven', 
                             'transmission', 'owner', 'location', 'car_age']
        for feature in required_features:
            if feature not in df.columns and feature != 'car_age':
                df[feature] = ''

        if preprocessor:
            processed_data = preprocessor.transform(df)
            return processed_data
        else:
            return df.select_dtypes(include=[np.number]).fillna(0).values
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['make', 'model', 'year', 'fuel', 'kms_driven', 
                           'transmission', 'owner', 'location']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        try:
            data['year'] = int(data['year'])
            data['kms_driven'] = float(data['kms_driven'])
        except ValueError:
            return jsonify({'error': 'Invalid numeric values'}), 400

        if model:
            processed_data = preprocess_input(data)
            prediction = model.predict(processed_data)
            predicted_price = float(prediction[0])
        else:
            base_price = 500000
            current_year = datetime.now().year
            age = current_year - data['year']
            depreciation = 0.15 * age
            km_factor = max(0.3, 1 - (data['kms_driven'] / 200000))
            brand_multiplier = 1.3 if data['make'] in ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi'] else 1.0
            fuel_multiplier = {'Petrol': 1.0, 'Diesel': 1.1, 'Electric': 1.5, 'CNG': 0.9, 'LPG': 0.85}.get(data['fuel'], 1.0)
            predicted_price = base_price * (1 - depreciation) * km_factor * brand_multiplier * fuel_multiplier
            predicted_price = max(50000, predicted_price)

        formatted_price = f"₹{predicted_price/100000:.1f} Lakhs" if predicted_price >= 100000 else f"₹{predicted_price:,.0f}"
        confidence = f"± ₹{predicted_price * 0.15:,.0f}"
        market_analysis = [
            f"Car age: {datetime.now().year - data['year']} years affects pricing significantly",
            f"High mileage vehicles (>{data['kms_driven']:,} km) typically see reduced values",
            f"{data['fuel']} vehicles have specific market demand patterns",
            f"{data['location']} market conditions influence final pricing"
        ]
        result = {
            'predicted_price': formatted_price,
            'confidence': confidence,
            'market_analysis': market_analysis,
            'raw_price': predicted_price
        }

        db_instance.log_prediction(data, result)
        logger.info(f"Prediction: {formatted_price} for {data['make']} {data['model']}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/car-makes', methods=['GET'])
def get_car_makes():
    try:
        if dataset_info and 'makes' in dataset_info:
            makes = dataset_info['makes']
        else:
            makes = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra',
                     'Ford', 'Volkswagen', 'Chevrolet', 'Nissan', 'Renault',
                     'BMW', 'Mercedes-Benz', 'Audi', 'Skoda', 'Fiat']
        return jsonify({'makes': sorted(makes)})
    except Exception as e:
        logger.error(f"Fetch makes error: {e}")
        return jsonify({'error': 'Failed to fetch car makes'}), 500

@app.route('/car-models/<make>', methods=['GET'])
def get_car_models(make):
    try:
        if dataset_info and 'models' in dataset_info and make in dataset_info['models']:
            models = dataset_info['models'][make]
        else:
            model_mapping = {
                'Maruti': ['Swift', 'Baleno', 'Alto', 'Wagon R', 'Dzire'],
                'Hyundai': ['i20', 'Creta', 'Verna'],
                'Honda': ['City', 'Amaze', 'Jazz'],
                'Toyota': ['Innova', 'Fortuner', 'Corolla'],
                'Tata': ['Nexon', 'Harrier'],
                'Mahindra': ['XUV500', 'Scorpio'],
            }
            models = model_mapping.get(make, ['Model 1', 'Model 2'])
        return jsonify({'models': sorted(models)})
    except Exception as e:
        logger.error(f"Fetch models error for {make}: {e}")
        return jsonify({'error': 'Failed to fetch car models'}), 500

@app.route('/locations', methods=['GET'])
def get_locations():
    try:
        if dataset_info and 'locations' in dataset_info:
            locations = dataset_info['locations']
        else:
            locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune']
        return jsonify({'locations': sorted(locations)})
    except Exception as e:
        logger.error(f"Fetch locations error: {e}")
        return jsonify({'error': 'Failed to fetch locations'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = db_instance.get_prediction_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Fetch stats error: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500

@app.route('/recent-predictions', methods=['GET'])
def get_recent_predictions():
    try:
        recent = db_instance.get_recent_predictions()
        return jsonify({'recent_predictions': recent})
    except Exception as e:
        logger.error(f"Fetch recent predictions error: {e}")
        return jsonify({'error': 'Failed to fetch recent predictions'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'database_connected': db_instance.db is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    load_model_and_preprocessor()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
