import lightgbm as lgb
import pandas as pd
import logging
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO)

def train_model(df, config):
    """
    Trains a LightGBM model. LGBM is chosen because it natively 
    learns the best split for missing values (handling the 25% constraint).
    """
    logging.info(f"Training model with provided DataFrame of shape {df.shape}")
    
    # Select features based on config
    features = config['features']['numeric'] + config['features']['categorical'] + config['features']['derived']
    target = config['features']['target']
    
    X = df[features]
    y = df[target]
     
    train_data = lgb.Dataset(X, label=y, categorical_feature=config['features']['categorical'])
    
    logging.info("Training Global LightGBM Model...")
    params = config['model_params']
    
    gbm = lgb.train(params, train_data, num_boost_round=100)
    
    model_path = config['paths']['model_path']
    gbm.save_model(model_path)
    logging.info(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    cfg = load_config()
    train_model(cfg['paths']['processed_data'], cfg)