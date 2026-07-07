import pandas as pd
import logging
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Transforms raw logs into model-ready features.
    Handles the 25% missing logs constraint by flagging them.
    """
    logging.info("Starting data preprocessing...")
    
   
    df['is_log_missing'] = df['memory_usage'].isnull().astype(int)
    
   
    numeric_cols = ['memory_usage', 'disk_io', 'network_latency']
    df[numeric_cols] = df.groupby('server_id')[numeric_cols].ffill()
    
    
    df['memory_usage_1h_avg'] = df['memory_usage'] # Placeholder for rolling logic
    df['disk_io_1h_max'] = df['disk_io']
    df['network_latency_1h_avg'] = df['network_latency']
   
    for col in config['features']['categorical']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    logging.info("Data preprocessing completed.")
    return df