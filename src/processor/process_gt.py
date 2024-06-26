from config.config import configure_logging
from src.utils.helpers import store_processed_data
import logging
import json
import os
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from config.config_paths import PUNCTUATION_LIST, DATA_RAW, DATA_PROCESSED

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Load and process GW data ground truth
def load_GW_gt() -> None:
    """
    Load, process GW dataset ground truth, and then store the data in data/processed.
    """
    # Load the punctuation abbreviation dictionary
    config_path = PUNCTUATION_LIST
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    punctuation_abbrev = config["punctuation_abbrev"]
    symbol_replacement = config["GW_symbol_replacement"]
    # Load the GW ground truth raw data
    gt_path = os.path.join(DATA_RAW, "GW", "ground_truth", "transcription.txt")
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
    destination_folder = os.path.join(DATA_PROCESSED, "GW", "ground_truth")
    store_processed_data("GW_gt", GW_gt, destination_folder)
    return

def load_jh_gt() -> None:
    """
    Load, process JH dataset ground truth, and then store the data in data/processed.
    """
    data = dict()

    # Reading data with a custom format
    with open(os.path.join(DATA_RAW, "jh", "transcription.txt"), 'r') as text_file:
        for line in text_file:
            key, value = line.strip().split(': ', 1)
            data[key] = value

    destination_folder = os.path.join(DATA_PROCESSED, "jh", "ground_truth")
    store_processed_data("jh_gt", data, destination_folder)
    return

# Load and process IAM data ground truth
def load_IAM_gt() -> None:
    """
    Load, process IAM dataset ground truth, and then store the data in data/processed.
    """
    IAM_gt = dict()
    # Parse the XML file
    gt_root_path = os.path.join(DATA_RAW, "IAM", "ground_truth")
    for filename in os.listdir(gt_root_path):
        file_path = os.path.join(gt_root_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        handwritten_part = root.find(".//handwritten-part")
        if handwritten_part is not None:
            for line_element in handwritten_part.findall(".//line"):
                # Access text and id content of the <line> element
                line_text = line_element.get("text")
                line_id = line_element.get("id")
                IAM_gt[line_id] = line_text
    destination_folder = os.path.join(DATA_PROCESSED, "IAM", "ground_truth")
    store_processed_data("IAM_gt", IAM_gt, destination_folder)
    return

def load_bullinger_gt() -> None:
    """
    Load, process Bullinger dataset ground truth, and then store the data in data/processed.
    """
    bullinger_gt = dict()
    root_path = os.path.join(DATA_RAW, "Bullinger")
    # Open the ZIP file
    for mode in ["train", "val", "test"]:
        for zip_filename in os.listdir(os.path.join(root_path, mode)):
            if zip_filename.endswith(".zip"):
                with ZipFile(os.path.join(root_path, mode, zip_filename), "r") as zip_file:
                    # List the contents of the ZIP file
                    file_list = zip_file.namelist()
                    file_list = [filename for filename in file_list if filename.endswith(".txt") and "de" in filename.split(os.path.sep)]
                    for filename in file_list:
                        with zip_file.open(filename) as gt:
                            key = mode+"_"+os.path.splitext(filename)[0]
                            bullinger_gt[key] = gt.read().decode("utf-8")
    destination_folder = os.path.join(DATA_PROCESSED, "Bullinger", "ground_truth")
    store_processed_data("bullinger_gt", bullinger_gt, destination_folder)
    return

def load_icfhr_gt() -> None:
    """
    Load ground truth data for the ICFHR 2016 dataset and store it in a processed format.

    The function reads XML files containing ground truth information for training, validation, and test sets
    and extracts text line coordinates and Unicode values. The data is then stored in a dictionary with keys
    in the format "mode__textline_id__file_id" and values as Unicode text.

    The processed data is stored in a destination folder using the `store_processed_data` function.

    Notes:
    - The XML files should follow the PAGE XML format (http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15).
    - The processed data is stored in the "icfhr_gt" dictionary.
    - The keys in the dictionary are in the format "mode__textline_id__file_id".
    - The values in the dictionary are Unicode text corresponding to the extracted text from the XML files.
    """
    icfhr_gt = dict()
    root_path = os.path.join(DATA_RAW, "ICFHR_2016", "ground_truth")
    for mode in ["train", "val", "test"]:
        files = os.listdir(os.path.join(root_path, mode))
        files = [file for file in files if file != ".DS_Store"]
        for file in files:
            path = os.path.join(root_path, mode, file)
            # Extracting coordinates
            tree = ET.parse(path)
            root = tree.getroot()
            # Get coordinate points and Unicode from TextLine elements
            for textline in root.findall(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine"):
                # Extract id and coordinate points from the xml file
                textline_id = textline.get("id")
                unicode_value = textline.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextEquiv/{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Unicode").text
                key = mode+"__"+textline_id+"__"+file.replace(".xml", "")
                icfhr_gt[key] = unicode_value
    destination_folder = os.path.join(DATA_PROCESSED, "ICFHR_2016", "ground_truth")
    store_processed_data("icfhr_gt", icfhr_gt, destination_folder)
    return

# Load ground truth depending on the input of the data name
def process_gt() -> None:
    """
    # TODO Fill German dataset in
    Process the ground truth of GW, IAM and "" datasets and store them data/processed/.
    """
    logger.info("Start processing GW raw ground truth data...")
    load_GW_gt()
    logger.info("Start processing JH raw ground truth data...")
    load_jh_gt()
    logger.info("Start processing IAM raw ground truth data...")
    load_IAM_gt()
    logger.info("Start processing Bullinger raw ground truth data...")
    load_bullinger_gt()
    logger.info("Start processing ICFHR 2016 raw ground truth data...")
    load_icfhr_gt()
    logger.info("All ground truth data processed and is stored in data/processed/.")
    return

if __name__=="__main__":
    # process_gt()
    load_jh_gt()
