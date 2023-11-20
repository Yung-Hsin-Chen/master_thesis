# Load batch size from the YAML file
with open(os.path.join(".", "config", "config_const.yaml"), "r") as config_file:
    config_data = yaml.safe_load(config_file)
batch_size = config_data["training"]["batch_size"]