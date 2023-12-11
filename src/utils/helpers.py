import os
import json
import configparser
from typing import List, Tuple
from config.config_paths import DATA_PROCESSED, CONFIG_CONST

# Store processed data dictionary into HDF5
def store_processed_data(data_name: str, data: dict, path: str) -> None:
    """
    Store the processed data dictionary into HDF5 in data/processed.

    Parameters
    -----------
    data_name: str
        # TODO add German dataset here
        The name of the data, can be "GW", "IAM" or "" with "_gt" or "_image" as the ending.
    data: dict
        The keys are the indices, while the values are the corresponding ground truth text or image path.
    path: str
        The destination folder to store the data

    Returns
    --------
    None
    """
    # Ensure the folder exists; create it if it doesn't
    os.makedirs(path, exist_ok=True)
    # Store the processed data dictionary as h5
    file_name = os.path.join(path, data_name+".json")
    with open(file_name, "w") as json_file:
        json.dump(data, json_file)
    return

def get_config_const(*config_keys: Tuple[str, str, type]) -> List:  
    # Get batch size and other values based on the provided keys
    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_CONST)

    values = []
    for key in config_keys:
        section, option, data_type = key
        try:
            # Attempt to get the value from the config file
            value_str = config_parser.get(section, option)
            
            # Convert the string value to the specified data type
            converted_value = data_type(value_str)
            values.append(converted_value)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            # Handle errors, e.g., section or option not found, or conversion error
            print(f"Error reading config: {e}")
            values.append(None)  # or some default value

    return values if len(values) > 1 else values[0]

def load_data() -> dict:
    """
    Load data dictionaries from json files as dictionaries.

    Returns
    --------
    {"en_image": GW_image, "en_gt": GW_gt}: dict
        The keys are the name of the dataset, and the values are the data in dictionaries.
        The data dictionaries have their file names as keys and data contents as values.
    """
    # Load ground truth data
    file_path = os.path.join(DATA_PROCESSED, "GW", "ground_truth", "GW_gt.json")
    with open(file_path, "r") as json_file:
        GW_gt = json.load(json_file)
    file_path = os.path.join(DATA_PROCESSED, "IAM", "ground_truth", "IAM_gt.json")
    with open(file_path, "r") as json_file:
        IAM_gt = json.load(json_file)
    file_path = os.path.join(DATA_PROCESSED, "Bullinger", "ground_truth", "bullinger_gt.json")
    with open(file_path, "r") as json_file:
        bullinger_gt = json.load(json_file)
    file_path = os.path.join(DATA_PROCESSED, "ICFHR_2016", "ground_truth", "icfhr_gt.json")
    with open(file_path, "r") as json_file:
        icfhr_gt = json.load(json_file)
    # Load image data
    file_path = os.path.join(DATA_PROCESSED, "GW", "line_image", "GW_image.json")
    with open(file_path, "r") as json_file:
        GW_image = json.load(json_file)
    file_path = os.path.join(DATA_PROCESSED, "IAM", "line_image", "IAM_image.json")
    with open(file_path, "r") as json_file:
        IAM_image = json.load(json_file)
    file_path = os.path.join(DATA_PROCESSED, "Bullinger", "line_image", "bullinger_image.json")
    with open(file_path, "r") as json_file:
        bullinger_image = json.load(json_file)
    file_path = os.path.join(DATA_PROCESSED, "ICFHR_2016", "line_image", "icfhr_image.json")
    with open(file_path, "r") as json_file:
        icfhr_image = json.load(json_file)
    return {"GW_image": GW_image, "GW_gt": GW_gt,
            "IAM_image": IAM_image, "IAM_gt": IAM_gt,
            "bullinger_image": bullinger_image, "bullinger_gt": bullinger_gt,
            "icfhr_image": icfhr_image, "icfhr_gt": icfhr_gt}

# def index_mapping():
#     data = load_data()
#     gw_indices = list(data["GW_image"].keys())
#     gw_numbers = range(len(gw_indices))
#     gw_idx_to_num = dict(zip(gw_indices, gw_numbers))
#     gw_num_to_idx = dict(zip(gw_numbers, gw_indices))
#     iam_indices = list(data["IAM_image"].keys())
#     iam_numbers = range(len(iam_indices))
#     iam_idx_to_num = dict(zip(iam_indices, iam_numbers))
#     iam_num_to_idx = dict(zip(iam_numbers, iam_indices))
#     bullinger_indices = list(data["bullinger_image"].keys())
#     bullinger_numbers = range(len(bullinger_indices))
#     bullinger_idx_to_num = dict(zip(bullinger_indices, bullinger_numbers))
#     bullinger_num_to_idx = dict(zip(bullinger_numbers, bullinger_indices))
#     icfhr_indices = list(data["icfhr_image"].keys())
#     icfhr_numbers = range(len(icfhr_indices))
#     icfhr_idx_to_num = dict(zip(icfhr_indices, icfhr_numbers))
#     icfhr_num_to_idx = dict(zip(icfhr_numbers, icfhr_indices))
#     return {"gw": {"idx_to_num": gw_idx_to_num, "num_to_idx": gw_num_to_idx}, 
#             "iam": {"idx_to_num": iam_idx_to_num, "num_to_idx": iam_num_to_idx}, 
#             "bullinger": {"idx_to_num": bullinger_idx_to_num, "num_to_idx": bullinger_num_to_idx}, 
#             "icfhr": {"idx_to_num": icfhr_idx_to_num, "num_to_idx": icfhr_num_to_idx}}
