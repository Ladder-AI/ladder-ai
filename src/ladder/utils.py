from loguru import logger
import json 
import sys
import os 


def load_json(json_path:str):
    if not os.path.exists(json_path):
        logger.error(f"Json file {json_path} does not exist")
        sys.exit(1)
    with open(json_path, 'r') as f:
        return json.load(f)