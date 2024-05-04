from config.config import configure_logging
from src.utils.helpers import store_processed_data
import logging
import os
from zipfile import ZipFile
from src.processor.crop_image import crop_image
from config.config_paths import DATA_RAW, DATA_PROCESSED

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Load and process data image
def load_en_image(data_name: str) -> None:
    """
    Load, process English dataset image paths, and then store the paths in data/processed.
    """
    images = dict()
    folder = os.path.join(DATA_RAW, data_name[:data_name.find("_")], "line_image")
    files = os.listdir(folder)
    # Filter out specific files (e.g. .DS_Store)
    files = [file for file in files if file != ".DS_Store"]
    for file in files:
        images[os.path.splitext(file)[0]] = os.path.join(folder, file)
    destination_folder = os.path.join(DATA_PROCESSED, data_name[:data_name.find("_")], "line_image")
    print(images)
    store_processed_data(data_name, images, destination_folder)
    return

def load_bullinger_image() -> None:
    """
    Crop, load and process ICFHR 2016 dataset image paths, and then store the paths in data/processed.
    """
    bullinger_image = dict()
    root_path = os.path.join(DATA_RAW, "Bullinger")
    # Open the ZIP file
    for mode in ["train", "val", "test"]:
        for zip_filename in os.listdir(os.path.join(root_path, mode)):
            if zip_filename.endswith(".zip"):
                with ZipFile(os.path.join(root_path, mode, zip_filename), "r") as zip_file:
                    # List the contents of the ZIP file
                    file_list = zip_file.namelist()
                    file_list = [filename for filename in file_list if filename.endswith(".png") and "de" in filename.split(os.path.sep)]
                    for filename in file_list:
                        key = mode+"_"+os.path.splitext(filename)[0]
                        bullinger_image[key] = filename
    destination_folder = os.path.join(DATA_PROCESSED, "Bullinger", "line_image")
    store_processed_data("bullinger_image", bullinger_image, destination_folder)
    return

def load_icfhr_image() -> None:
    """
    Load, process ICFHR 2016 dataset image paths, and then store the paths in data/processed.
    """
    icfhr_image = dict()
    root_path = os.path.join(DATA_RAW, "ICFHR_2016", "page_image")
    for mode in ["train", "val", "test"]:
        files = os.listdir(os.path.join(root_path, mode))
        # Filter out specific files (e.g. .DS_Store)
        files = [file for file in files if file != ".DS_Store"]
        for file in files:
            input_path = os.path.join(root_path, mode, file)
            temp = crop_image(input_path, mode)
            icfhr_image.update(temp)
    destination_folder = os.path.join(DATA_PROCESSED, "ICFHR_2016", "line_image")
    store_processed_data("icfhr_image", icfhr_image, destination_folder)
    return

# Load images depending on the input of the data name
def process_image() -> None:
    """
    Process the images of GW, IAM and "Bullinger" datasets and store them data/processed/.
    """
    logger.info("Start processing English raw image data...")
    for data_name in ["GW_image", "IAM_image"]:
        load_en_image(data_name)
    logger.info("Start processing German raw image data...")
    load_bullinger_image()
    load_icfhr_image()
    logger.info("All image data processed and is stored in data/processed/.")
    return

if __name__=="__main__":
    # process_image()
    load_en_image("GW_image")
