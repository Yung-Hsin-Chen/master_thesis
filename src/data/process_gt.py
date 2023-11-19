from config.config import configure_logging
import logging
import json
import os
import xml.etree.ElementTree as ET
import h5py

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Store processed data dictionary into HDF5
def store_processed_data(data_name: str, data: dict) -> None:
    """
    Store the processed data dictionary into HDF5 in data/processed.

    Parameters
    -----------
    data_name: str
        The name of the data, can be "GW", "IAM" or "".
    data: dict
        The keys are the indices, while the values are the corresponding ground truth text.

    Returns
    --------
    None
    """
    # TODO include the GERMAN dataset name in docstring
    processed_path = os.path.join(".", "data", "processed", data_name, "ground_truth")
    # Ensure the folder exists; create it if it doesn't
    os.makedirs(processed_path, exist_ok=True)
    # Store the processed data dictionary as h5
    file_name = os.path.join(processed_path, data_name+"_gt.h5")
    with h5py.File(file_name, "w") as h5_file:
        for key, value in data.items():
            h5_file[key] = value
    return

# Load and process GW data ground truth
def load_GW_gt() -> None:
    """
    Load and Process GW dataset ground truth.

    Parameters
    -----------
    None

    Returns
    --------
    None
    """
    # Load the punctuation abbreviation dictionary
    config_path = os.path.join(".", "config", "config.json")
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    punctuation_abbrev = config["punctuation_abbrev"]
    symbol_replacement = config["GW_symbol_replacement"]
    # Load the GW ground truth raw data
    gt_path = os.path.join(".", "data", "raw", "GW", "ground_truth", "transcription.txt")
    with open(gt_path, "r") as file:
        GW_gt = file.readlines()
    GW_gt = {i[:6]: i[7:] for i in GW_gt}
    # Replace the punctuation abbreviations with the actual punctuation
    for page,line in GW_gt.items():
        for k,v in punctuation_abbrev.items():
            line = line.replace(k, v)
        for k,v in symbol_replacement.items():
            line = line.replace(k, v)
        GW_gt[page] = line.strip()
    store_processed_data("GW", GW_gt)
    return

# Load and process IAM data ground truth
def load_IAM_gt() -> None:
    """
    Load and Process IAM dataset ground truth.

    Parameters
    -----------
    None

    Returns
    --------
    None
    """
    IAM_gt = dict()
    # Parse the XML file
    gt_root_path = os.path.join(".", "data", "raw", "IAM", "ground_truth")
    for filename in os.listdir(gt_root_path):
        file_path = os.path.join(gt_root_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        handwritten_part = root.find('.//handwritten-part')
        if handwritten_part is not None:
            for line_element in handwritten_part.findall(".//line"):
                # Access text and id content of the <line> element
                line_text = line_element.get("text")
                line_id = line_element.get("id")
                IAM_gt[line_id] = line_text
    store_processed_data("IAM", IAM_gt)
    return

# Load ground truth depending on the input of the data name
def process_gt() -> None:
    logger.info("Start processing GW raw data...")
    load_GW_gt()

    logger.info("Start processing IAM raw data...")
    load_IAM_gt()
    # TODO Expand German dataset
    logger.info("Data processed and is stored in data/processed/.")
    return

if __name__=="__main__":
    # print(os.getcwd())
    process_gt()
