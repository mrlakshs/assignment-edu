import lightgbm as lgb
import pandas as pd
import time
import logging
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO)

def predict_server_failure(new_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Loads the trained model and performs low-latency batch inference.
    """
    model_path = config['paths']['model_path']
    logging.info("Loading model for inference...")
    
    start_time = time.time()
    gbm = lgb.Booster(model_file=model_path)
    
    features = config['features']['numeric'] + config['features']['categorical'] + config['features']['derived']
    X_new = new_data[features]
    
    # Generate probabilities
    predictions = gbm.predict(X_new)
    
    inference_time = time.time() - start_time
    logging.info(f"Scored {len(new_data)} servers in {inference_time:.4f} seconds.")
    
    if inference_time > 5.0:
        logging.warning("Inference time exceeded 5 seconds constraint!")
        
    new_data['failure_probability'] = predictions
    new_data['high_risk_flag'] = (predictions > 0.85).astype(int)
    
    return new_data[['server_id', 'failure_probability', 'high_risk_flag']]