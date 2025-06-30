# Used Car Price Prediction Backend

This is the Python Flask backend for the Used Car Price Prediction application. It provides ML-powered price predictions through a RESTful API.

## Features

- **Machine Learning Model**: Random Forest Regressor trained on 25,000+ car records
- **Real-time Predictions**: Instant price predictions with confidence intervals
- **RESTful API**: Clean endpoints for car data and predictions
- **Data Preprocessing**: Robust feature engineering and encoding
- **Market Analysis**: Contextual insights with predictions

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager

### 2. Installation

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model

Before running the server, you need to train the ML model:

```bash
python train_model.py
```

This will:
- Generate a sample dataset with 25,000 car records
- Train a Random Forest model
- Save the model, preprocessor, and dataset info

### 4. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Predict Car Price
- **POST** `/predict`
- **Body**: 
```json
{
  "make": "Hyundai",
  "model": "i20",
  "year": 2018,
  "fuel": "Petrol",
  "kms_driven": 45000,
  "transmission": "Manual",
  "owner": "First Owner",
  "location": "Pune"
}
```

### Get Car Makes
- **GET** `/car-makes`
- Returns list of available car manufacturers

### Get Car Models
- **GET** `/car-models/<make>`
- Returns models for a specific manufacturer

### Get Locations
- **GET** `/locations`
- Returns available cities/locations

### Health Check
- **GET** `/health`
- Returns server and model status

## Model Details

### Dataset Features
- **make**: Car manufacturer (Maruti, Hyundai, etc.)
- **model**: Specific car model
- **year**: Manufacturing year
- **fuel**: Fuel type (Petrol, Diesel, CNG, LPG, Electric)
- **kms_driven**: Total kilometers driven
- **transmission**: Manual or Automatic
- **owner**: Owner type (First, Second, Third, etc.)
- **location**: City where car is sold

### Model Performance
- **Algorithm**: Random Forest Regressor
- **Training MAE**: ~₹45,000
- **Testing MAE**: ~₹52,000
- **R² Score**: 0.85+

### Price Prediction Logic
1. **Base Price**: Determined by make and model
2. **Depreciation**: 12-15% per year based on brand
3. **Usage Impact**: Reduces value based on kilometers driven
4. **Feature Premiums**: Automatic transmission, fuel type adjustments
5. **Location Factor**: Metro cities have higher prices
6. **Owner Impact**: First owner commands premium

## File Structure

```
backend/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained model (generated)
├── preprocessor.pkl      # Data preprocessor (generated)
├── dataset_info.pkl      # Dataset metadata (generated)
├── data/
│   └── raw/
│       └── used_cars_large.csv  # Training dataset
└── README.md
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

- `FLASK_ENV`: Set to 'production' for production deployment
- `PORT`: Server port (default: 5000)

## Dataset Information

The model is trained on a synthetic but realistic dataset containing:
- 25,000+ car records
- 10 major Indian car manufacturers
- 50+ popular car models
- Price range: ₹50,000 to ₹50,00,000
- Years: 2005-2024

For production use, replace with real market data from sources like:
- Cars24, CarDekho, OLX APIs
- Kaggle automotive datasets
- Government vehicle registration data

## API Response Format

### Successful Prediction
```json
{
  "predicted_price": "₹5.2 Lakhs",
  "confidence": "± ₹25,000",
  "market_analysis": [
    "Car age: 6 years affects pricing significantly",
    "High mileage vehicles (>45,000 km) typically see reduced values",
    "Petrol vehicles have specific market demand patterns",
    "Pune market conditions influence final pricing"
  ],
  "raw_price": 520000
}
```

### Error Response
```json
{
  "error": "Missing required field: make"
}
```

## Troubleshooting

### Model Files Not Found
Run the training script first:
```bash
python train_model.py
```

### CORS Issues
The server includes CORS headers for frontend integration. If issues persist, check the frontend API URL configuration.

### Performance Optimization
- Use Redis for caching frequent predictions
- Implement batch prediction endpoints
- Add database logging for usage analytics