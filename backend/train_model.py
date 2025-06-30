import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset(n_samples=25000):
    """Create a realistic sample dataset for training"""
    logger.info(f"Creating sample dataset with {n_samples} records...")
    
    np.random.seed(42)
    
    # Car makes and their typical models
    car_data = {
        'Maruti': ['Swift', 'Baleno', 'Alto', 'Wagon R', 'Dzire', 'Vitara Brezza', 'Ertiga', 'Ciaz'],
        'Hyundai': ['i20', 'Creta', 'Verna', 'Grand i10', 'Santro', 'Venue', 'Elantra', 'Tucson'],
        'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'Civic', 'CR-V', 'Accord', 'BR-V'],
        'Toyota': ['Innova', 'Fortuner', 'Corolla', 'Camry', 'Etios', 'Yaris', 'Glanza', 'Urban Cruiser'],
        'Tata': ['Nexon', 'Harrier', 'Safari', 'Altroz', 'Tigor', 'Tiago', 'Hexa', 'Bolt'],
        'Mahindra': ['XUV500', 'Scorpio', 'Bolero', 'XUV300', 'Thar', 'KUV100', 'Marazzo', 'Alturas'],
        'Ford': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle', 'Mustang'],
        'Volkswagen': ['Polo', 'Vento', 'Tiguan', 'Passat', 'Jetta', 'Ameo'],
        'BMW': ['3 Series', '5 Series', 'X1', 'X3', 'X5', '7 Series'],
        'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLE']
    }
    
    locations = [
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Kolkata',
        'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur'
    ]
    
    fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
    transmission_types = ['Manual', 'Automatic']
    owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
    
    # Generate data
    data = []
    
    for _ in range(n_samples):
        make = np.random.choice(list(car_data.keys()))
        model = np.random.choice(car_data[make])
        year = np.random.randint(2005, 2024)
        fuel = np.random.choice(fuel_types, p=[0.4, 0.35, 0.1, 0.1, 0.05])
        transmission = np.random.choice(transmission_types, p=[0.7, 0.3])
        owner = np.random.choice(owner_types, p=[0.4, 0.35, 0.2, 0.05])
        location = np.random.choice(locations)
        
        # Generate realistic km based on age
        car_age = 2024 - year
        avg_km_per_year = np.random.normal(12000, 3000)
        kms_driven = max(500, int(car_age * avg_km_per_year + np.random.normal(0, 10000)))
        
        # Calculate realistic price
        base_prices = {
            'Maruti': 600000, 'Hyundai': 700000, 'Honda': 800000,
            'Toyota': 900000, 'Tata': 650000, 'Mahindra': 750000,
            'Ford': 700000, 'Volkswagen': 800000, 'BMW': 2500000,
            'Mercedes-Benz': 3000000
        }
        
        base_price = base_prices[make]
        
        # Apply depreciation
        depreciation_rate = 0.12 if make in ['BMW', 'Mercedes-Benz'] else 0.15
        price = base_price * (1 - depreciation_rate) ** car_age
        
        # Adjust for kilometers
        km_factor = max(0.4, 1 - (kms_driven / 300000))
        price *= km_factor
        
        # Fuel type adjustment
        fuel_multipliers = {'Petrol': 1.0, 'Diesel': 1.1, 'Electric': 1.4, 'CNG': 0.9, 'LPG': 0.85}
        price *= fuel_multipliers[fuel]
        
        # Transmission adjustment
        if transmission == 'Automatic':
            price *= 1.15
        
        # Owner adjustment
        owner_multipliers = {'First Owner': 1.0, 'Second Owner': 0.85, 'Third Owner': 0.7, 'Fourth & Above Owner': 0.6}
        price *= owner_multipliers[owner]
        
        # Location adjustment (metro cities are more expensive)
        metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Kolkata', 'Hyderabad']
        if location in metro_cities:
            price *= 1.1
        
        # Add some random noise
        price *= np.random.normal(1.0, 0.1)
        price = max(50000, int(price))  # Minimum price
        
        data.append({
            'make': make,
            'model': model,
            'year': year,
            'fuel': fuel,
            'kms_driven': kms_driven,
            'transmission': transmission,
            'owner': owner,
            'location': location,
            'price': price
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Dataset created with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    logger.info("Preprocessing data...")
    
    # Create derived features
    current_year = datetime.now().year
    df['car_age'] = current_year - df['year']
    
    # Define features and target
    feature_columns = ['make', 'model', 'year', 'fuel', 'kms_driven', 
                      'transmission', 'owner', 'location', 'car_age']
    
    X = df[feature_columns].copy()
    y = df['price'].copy()
    
    # Identify categorical and numerical columns
    categorical_features = ['make', 'model', 'fuel', 'transmission', 'owner', 'location']
    numerical_features = ['year', 'kms_driven', 'car_age']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', LabelEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Note: LabelEncoder in ColumnTransformer needs special handling
    # Let's create a custom preprocessing approach
    X_processed = X.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_processed[numerical_features] = scaler.fit_transform(X_processed[numerical_features])
    
    # Create a combined preprocessor
    class CustomPreprocessor:
        def __init__(self, label_encoders, scaler, categorical_features, numerical_features):
            self.label_encoders = label_encoders
            self.scaler = scaler
            self.categorical_features = categorical_features
            self.numerical_features = numerical_features
        
        def transform(self, X):
            X_transformed = X.copy()
            
            # Handle categorical features
            for col in self.categorical_features:
                if col in X_transformed.columns:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X_transformed[col] = X_transformed[col].astype(str)
                    
                    # Map unseen categories to 0 or most frequent category
                    mask = X_transformed[col].isin(le.classes_)
                    X_transformed.loc[~mask, col] = le.classes_[0]  # Use first class for unseen
                    X_transformed[col] = le.transform(X_transformed[col])
            
            # Handle numerical features
            if set(self.numerical_features).issubset(X_transformed.columns):
                X_transformed[self.numerical_features] = self.scaler.transform(X_transformed[self.numerical_features])
            
            return X_transformed
    
    custom_preprocessor = CustomPreprocessor(label_encoders, scaler, categorical_features, numerical_features)
    
    return X_processed, y, custom_preprocessor

def train_model(X, y):
    """Train the Random Forest model"""
    logger.info("Training Random Forest model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    logger.info(f"Training MAE: ₹{train_mae:,.0f}")
    logger.info(f"Testing MAE: ₹{test_mae:,.0f}")
    logger.info(f"Training R²: {train_r2:.3f}")
    logger.info(f"Testing R²: {test_r2:.3f}")
    
    return model

def save_model_and_info(model, preprocessor, df):
    """Save model, preprocessor, and dataset info"""
    logger.info("Saving model and preprocessor...")
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Create dataset info for API endpoints
    dataset_info = {
        'makes': df['make'].unique().tolist(),
        'models': {make: df[df['make'] == make]['model'].unique().tolist() 
                  for make in df['make'].unique()},
        'locations': df['location'].unique().tolist(),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'total_records': len(df)
    }
    
    with open('dataset_info.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)
    
    logger.info("Model, preprocessor, and dataset info saved successfully!")

def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline...")
    
    # Create or load dataset
    try:
        # Try to load existing dataset
        df = pd.read_csv('data/raw/used_cars_large.csv')
        logger.info(f"Loaded existing dataset with {len(df)} records")
    except FileNotFoundError:
        logger.info("Creating sample dataset...")
        df = create_sample_dataset(25000)
        # Save the dataset
        import os
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/used_cars_large.csv', index=False)
        logger.info("Sample dataset saved to data/raw/used_cars_large.csv")
    
    # Preprocess data
    X, y, preprocessor = preprocess_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save everything
    save_model_and_info(model, preprocessor, df)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()