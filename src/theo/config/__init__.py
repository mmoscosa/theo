import os
import yaml

def load_yaml_config(filename):
    config_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(config_dir, filename)
    with open(path, 'r') as f:
        return yaml.safe_load(f) 