from pymongo import MongoClient
from datetime import datetime
import numpy as np

class DataBaseHandler:
    def __init__(self):
        # 2. Add 'self.' prefix to store connection details
        self.CONNECTION_STRING = "mongodb+srv://admin:helloadmin123@fish.caqok5c.mongodb.net/?appName=Fish"
        self.client = None
        self.fish_data_collection = None
        
        self.connect() # Call the connection logic

    def _convert_to_mongodb_types(self, data):
        """
        Convert numpy types and other non-serializable types to MongoDB-compatible types.
        """
        if isinstance(data, dict):
            return {k: self._convert_to_mongodb_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_mongodb_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif data is None:
            return None
        elif isinstance(data, (str, int, float, bool, datetime)):
            return data
        else:
            # Try to convert to string as last resort
            try:
                return str(data)
            except:
                return None

    def connect(self):
        """Initializes and stores the MongoDB connection and collection."""
        try:
            # Connect to MongoDB Atlas
            client = MongoClient(self.CONNECTION_STRING, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')  # Test the connection
            
            # Store client and collection on the instance
            self.client = client
            self.fish_data_collection = client.fish_tracking_db.fish_data
            
            print(" Successfully connected to MongoDB Atlas!")
        except Exception as e:
            print(f" Error connecting to MongoDB Atlas: {e}")
            self.client = None
            self.fish_data_collection = None
    
    def save_data_to_db(self,fish_id, size, distance_traveled_m, is_active, location_m, status):
        """
        Saves or updates a fish's record in MongoDB.
        """
        if self.fish_data_collection is None:  # If DB is not connected
            print("Database connection not available. Skipping save.")  # Print warning
            return  # Exit function
        try:

             # Convert all values to MongoDB-compatible types
            fish_id_converted = self._convert_to_mongodb_types(fish_id)
            size_converted = self._convert_to_mongodb_types(size)
            distance_converted = self._convert_to_mongodb_types(distance_traveled_m)
            location_converted = self._convert_to_mongodb_types(location_m)

            fish_document = {
                "fish_id": fish_id_converted,
                "size": size_converted,
                "distance_traveled_meters": distance_converted,
                "last_updated": datetime.now(),
                "is_active": bool(is_active),  # Ensure boolean
                "current_location_millimeters": location_converted,
                "status": str(status)  # Ensure string
            }
            
            # Clean up document (remove None values)
            fish_document = {k: v for k, v in fish_document.items() if v is not None}

            self.fish_data_collection.update_one(  # Update or insert document
                {"fish_id": fish_id_converted},
                {"$set": fish_document},
                upsert=True
            )
        except Exception as e:
            print(f"An error occurred while saving data: {e}")  # Print error

    def get_last_known_distance(self,fish_id):
        """
        Fetches the last known total distance traveled for a fish from MongoDB.
        """
        if self.fish_data_collection is None:  # If DB is not connected
            return 0.0  # Return zero
        try:
            fish_id_converted = self._convert_to_mongodb_types(fish_id)
            document = self.fish_data_collection.find_one({"fish_id": fish_id_converted})  # Query DB for fish_id
            if document and "distance_traveled_meters" in document:  # If found and has distance
                return document["distance_traveled_meters"]  # Return stored distance
            return 0.0  # If not found, return zero
        except Exception as e:
            print(f"Error fetching distance for fish {fish_id_converted}: {e}")  # Print error
            return 0.0  # Return zero on error
    
    def is_connected(self):
        """Returns connection status."""
        return self.connected
    
    def close(self):
        """Close the database connection."""
        if self.client:
            try:
                self.client.close()
                self.connected = False
                print("Database connection closed.")
            except Exception as e:
                print(f"Error closing database connection: {e}")