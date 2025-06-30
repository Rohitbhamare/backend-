import os
from pymongo import MongoClient
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI')
            if not mongodb_uri:
                logger.warning("MONGODB_URI not found. Database features disabled.")
                return
            
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.car_price_predictor
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
    
    def log_prediction(self, car_data, prediction_result):
        """Log prediction to database"""
        if not self.db:
            return
        
        try:
            prediction_log = {
                'timestamp': datetime.utcnow(),
                'car_data': car_data,
                'prediction': prediction_result,
                'ip_address': None  # Can be added from request
            }
            
            self.db.predictions.insert_one(prediction_log)
            logger.info("Prediction logged to database")
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def get_prediction_stats(self):
        """Get prediction statistics"""
        if not self.db:
            return {}
        
        try:
            total_predictions = self.db.predictions.count_documents({})
            
            # Most predicted brands
            brand_pipeline = [
                {"$group": {"_id": "$car_data.make", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ]
            top_brands = list(self.db.predictions.aggregate(brand_pipeline))
            
            # Average predicted prices by year
            year_pipeline = [
                {"$group": {
                    "_id": "$car_data.year", 
                    "avg_price": {"$avg": "$prediction.raw_price"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": -1}},
                {"$limit": 10}
            ]
            price_by_year = list(self.db.predictions.aggregate(year_pipeline))
            
            return {
                'total_predictions': total_predictions,
                'top_brands': top_brands,
                'price_by_year': price_by_year
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        if not self.db:
            return []
        
        try:
            recent = self.db.predictions.find(
                {},
                {'car_data': 1, 'prediction.predicted_price': 1, 'timestamp': 1}
            ).sort('timestamp', -1).limit(limit)
            
            return list(recent)
            
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []

# Global database instance
db_instance = Database()