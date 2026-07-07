import yaml
import logging

def load_config(config_path="configs/config.yaml"):
    """
    Safely loads the YAML configuration file.
    Prevents hardcoding paths and model names in the main execution scripts.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}. Check your path.")
        raise
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        raise