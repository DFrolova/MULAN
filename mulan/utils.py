import yaml


# Function to load yaml configuration file
def load_config(config_fname):
    with open(config_fname) as file:
        config = yaml.safe_load(file)
    return config