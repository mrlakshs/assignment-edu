import yaml
import logging

def load_config(config_path="configs/config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        raise