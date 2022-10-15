import yaml
from yaml.loader import SafeLoader

import os

PACKAGE_ROOT = os.path.dirname(os.path.abspath("train.py"))
CONFIG_FILE_PATH = PACKAGE_ROOT + "/" + "config.yaml"


def get_config_from_yaml(config_path):
    if config_path:
        with open(config_path, "r") as conf_file:
            parsed_config = yaml.load(conf_file, Loader=SafeLoader)
        return(parsed_config)
    raise OSError(f"Did not find config file at path: {CONFIG_FILE_PATH}")


config = get_config_from_yaml(CONFIG_FILE_PATH)
