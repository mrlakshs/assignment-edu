from src.config_loader import load_config
from src.data_pipeline import preprocess_data
from src.train import train_model
from src.predict import predict_server_failure
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Starting Pipeline ---")
    config = load_config()
    print("Config loaded!")
    
    print("Loading raw data...")
    raw_df = pd.read_csv(config['paths']['raw_data'])
    
    print("Preprocessing data...")
    data = preprocess_data(raw_df, config)
    
    print("Training LightGBM model...")
    model = train_model(data, config)
    
    print("Running predictions...")
    results = predict_server_failure(data, config)
    
    print(f"\nFinal Prediction Results:\n{results}")
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()