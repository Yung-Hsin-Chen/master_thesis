from config.config import configure_logging
from src.utils.helpers import store_processed_data
import logging
import os

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Load and process data image
def load_image_to_tensor(data_name: str) -> None:
    """
    Load, process dataset images to tensor, and then store the data in data/processed.

    Parameters
    -----------
    data_name: str

    Returns
    --------
    None
    """
    images = dict()
    folder = os.path.join(".", "data", "raw", data_name[:data_name.find("_")], "line_image")
    files = os.listdir(folder)
    # Filter out specific files (e.g. .DS_Store)
    files = [file for file in files if file != ".DS_Store"]
    for file in files:
        images[os.path.splitext(file)[0]] = os.path.join(folder, file)
    destination_folder = os.path.join(".", "data", "processed", data_name[:data_name.find("_")], "line_image")
    store_processed_data(data_name, images, destination_folder)
    return

# Load images depending on the input of the data name
def process_image() -> None:
    """
    # TODO Fill German dataset in
    Process the images of GW, IAM and "" datasets and store them data/processed/.

    Parameters
    -----------
    None

    Returns
    --------
    None
    """
    logger.info("Start processing all raw image data...")
    # TODO Expand German dataset
    for data_name in ["GW_image", "IAM_image"]:
        load_image_to_tensor(data_name)
    logger.info("All image data processed and is stored in data/processed/.")
    return

if __name__=="__main__":
    process_image()
