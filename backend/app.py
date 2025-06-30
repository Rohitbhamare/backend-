from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from database import db_instance

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None
dataset_info = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, preprocessor, dataset_info
    
    try:
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
            
        # Load dataset info for dropdowns
        with open('dataset_info.pkl', 'rb') as f:
            dataset_info = pickle.load(f)
            
        logger.info("Model and preprocessor loaded successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.info("Please run the training script first to generate model files")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Add derived features
        current_year = datetime.now().year
        df['car_age'] = current_year - df['year']
        
        # Ensure all required columns are present
        required_features = ['make', 'model', 'year', 'fuel', 'kms_driven', 
                           'transmission', 'owner', 'location', 'car_age']
        
        for feature in required_features:
            if feature not in df.columns and feature != 'car_age':
                df[feature] = ''
        
        # Apply preprocessor if available
        if preprocessor:
            processed_data = preprocessor.transform(df)
            return processed_data
        else:
            # Fallback processing
            return df.select_dtypes(include=[np.number]).fillna(0).values
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict_price():
    """Predict car price endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['make', 'model', 'year', 'fuel', 'kms_driven', 
                          'transmission', 'owner', 'location']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert numeric fields
        try:
            data['year'] = int(data['year'])
            data['kms_driven'] = float(data['kms_driven'])
        except ValueError:
            return jsonify({'error': 'Invalid numeric values'}), 400
        
        if not model:
            # Fallback prediction logic
            base_price = 500000  # Base price in rupees
            
            # Adjust based on year (depreciation)
            current_year = datetime.now().year
            age = current_year - data['year']
            depreciation = 0.15 * age  # 15% per year
            
            # Adjust based on kilometers
            km_factor = max(0.3, 1 - (data['kms_driven'] / 200000))  # Reduce price based on usage
            
            # Brand premium
            premium_brands = ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi']
            brand_multiplier = 1.3 if data['make'] in premium_brands else 1.0
            
            # Fuel type adjustment
            fuel_multipliers = {
                'Petrol': 1.0,
                'Diesel': 1.1,
                'Electric': 1.5,
                'CNG': 0.9,
                'LPG': 0.85
            }
            fuel_multiplier = fuel_multipliers.get(data['fuel'], 1.0)
            
            predicted_price = base_price * (1 - depreciation) * km_factor * brand_multiplier * fuel_multiplier
            predicted_price = max(50000, predicted_price)  # Minimum price
            
        else:
            # Use trained model
            processed_data = preprocess_input(data)
            prediction = model.predict(processed_data)
            predicted_price = float(prediction[0])
        
        # Format price
        if predicted_price >= 100000:
            formatted_price = f"₹{predicted_price/100000:.1f} Lakhs"
        else:
            formatted_price = f"₹{predicted_price:,.0f}"
        
        # Calculate confidence interval (approximate)
        confidence_range = predicted_price * 0.15  # ±15%
        confidence = f"± ₹{confidence_range:,.0f}"
        
        # Generate market analysis
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
        
        # Log prediction to database
        db_instance.log_prediction(data, result)
        
        logger.info(f"Prediction made: {formatted_price} for {data['make']} {data['model']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/car-makes', methods=['GET'])
def get_car_makes():
    """Get available car makes"""
    try:
        if dataset_info and 'makes' in dataset_info:
            makes = dataset_info['makes']
        else:
            # Fallback data
            makes = [
                'Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra', 
                'Ford', 'Volkswagen', 'Chevrolet', 'Nissan', 'Renault',
                'BMW', 'Mercedes-Benz', 'Audi', 'Skoda', 'Fiat'
            ]
        
        return jsonify({'makes': sorted(makes)})
    
    except Exception as e:
        logger.error(f"Error fetching makes: {e}")
        return jsonify({'error': 'Failed to fetch car makes'}), 500

@app.route('/car-models/<make>', methods=['GET'])
def get_car_models(make):
    """Get available models for a specific make"""
    try:
        if dataset_info and 'models' in dataset_info and make in dataset_info['models']:
            models = dataset_info['models'][make]
        else:
            # Fallback data
            model_mapping = {
                'Maruti': ['Swift', 'Baleno', 'Alto', 'Wagon R', 'Dzire', 'Vitara Brezza', 'Ertiga', 'Ciaz'],
                'Hyundai': ['i20', 'Creta', 'Verna', 'Grand i10', 'Santro', 'Venue', 'Elantra', 'Tucson'],
                'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'Civic', 'CR-V', 'Accord', 'BR-V'],
                'Toyota': ['Innova', 'Fortuner', 'Corolla', 'Camry', 'Etios', 'Yaris', 'Glanza', 'Urban Cruiser'],
                'Tata': ['Nexon', 'Harrier', 'Safari', 'Altroz', 'Tigor', 'Tiago', 'Hexa', 'Bolt'],
                'Mahindra': ['XUV500', 'Scorpio', 'Bolero', 'XUV300', 'Thar', 'KUV100', 'Marazzo', 'Alturas']
            }
            models = model_mapping.get(make, ['Model 1', 'Model 2', 'Model 3'])
        
        return jsonify({'models': sorted(models)})
    
    except Exception as e:
        logger.error(f"Error fetching models for {make}: {e}")
        return jsonify({'error': 'Failed to fetch car models'}), 500

@app.route('/locations', methods=['GET'])
def get_locations():
    """Get available locations"""
    try:
        if dataset_info and 'locations' in dataset_info:
            locations = dataset_info['locations']
        else:
            # Fallback data
            locations = [
                'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Kolkata',
                'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur',
                'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam',
                'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana'
            ]
        
        return jsonify({'locations': sorted(locations)})
    
    except Exception as e:
        logger.error(f"Error fetching locations: {e}")
        return jsonify({'error': 'Failed to fetch locations'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics"""
    try:
        stats = db_instance.get_prediction_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500

@app.route('/recent-predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions"""
    try:
        recent = db_instance.get_recent_predictions()
        return jsonify({'recent_predictions': recent})
    except Exception as e:
        logger.error(f"Error fetching recent predictions: {e}")
        return jsonify({'error': 'Failed to fetch recent predictions'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
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