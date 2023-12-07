from PIL import Image
from config.config import configure_logging
import logging
import os
import xml.etree.ElementTree as ET
from config.config_paths import DATA_RAW

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

def get_coordinate(input_path) -> tuple:
    """
    Extract text line coordinates from a PAGE XML file corresponding to a given image.

    Parameters
    -----------
    input_path: str
        Path to the image file (JPG format) for which coordinates are to be extracted.
        The function assumes that a corresponding XML file exists in the same directory and follows
        the PAGE XML format (http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15).

    Returns
    --------
    list
        A list of tuples, where each tuple contains the textline_id and the rectangular coordinates
        (left, top, right, bottom) of a text line in the image.

    Notes:
    - The function replaces the image file extension with ".xml" and adjusts the path to the ground truth folder.
    - The coordinates are extracted from the "Coords" element in the PAGE XML file.
    - The result is a list of tuples, where each tuple contains the textline_id and corresponding rectangular coordinates.
    - The coordinates are represented as (left, top, right, bottom).

    """
    coordinate_path = input_path.replace(".JPG", ".xml").replace("page_image", "ground_truth")
    # Extracting coordinates
    tree = ET.parse(coordinate_path)
    root = tree.getroot()
    # Get coordinate points and Unicode from TextLine elements
    result = []
    for textline in root.findall(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine"):
        # Extract id and coordinate points from the xml file
        textline_id = textline.get("id")
        coords_points = textline.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords").get("points")
        # Get rectangular coordinates from the polyfon points
        coordinates = [tuple(map(int, point.split(","))) for point in coords_points.split()]
        left = min(point[0] for point in coordinates)
        top = min(point[1] for point in coordinates)
        right = max(point[0] for point in coordinates)
        bottom = max(point[1] for point in coordinates)
        result.append((textline_id, (left, top, right, bottom)))
    return result


def crop_image(input_path, mode):
    """
    Crop the input image based on text line coordinates and save the cropped images.

    Parameters
    -----------
    input_path: str
        Path to the input image file (JPG format).
    mode: str
        Mode indicating the dataset (e.g., "train", "val", "test").

    Returns
    --------
    dict
        A dictionary mapping filename (mode__textline_id__original_filename) to the corresponding cropped image path.

    Notes:
    - The function uses the `get_coordinate` function to obtain text line coordinates from the corresponding XML file.
    - The image is cropped based on the text line coordinates (left, top, right, bottom).
    - Cropped images are saved in the "line_image" folder within the same directory as the input image.
    - The filename format for the cropped images is "mode__textline_id__original_filename".
    - The function returns a dictionary mapping the filename to the corresponding cropped image path.

    """
    icfhr_image = dict()
    # Open the image file
    img = Image.open(input_path)

    # Crop the image based on coordinates (left, top, right, bottom)
    coordinates = get_coordinate(input_path)
    # print(coordinates)
    for textline_id, coordinate in coordinates:
        cropped_img = img.crop(coordinate)
        # Save the cropped image
        output_path = os.path.dirname(input_path).replace("page_image", "line_image")
        filename = mode+"__"+textline_id+"__"+os.path.basename(input_path)
        output_path = os.path.join(output_path, filename)
        # output_path = input_path.replace("page_image", "line_image")
        cropped_img.save(output_path)
        icfhr_image[filename.replace(".JPG", "")] = output_path
    return icfhr_image


# if __name__=="__main__":
#     logger.info("Start cropping ICFHR 2016 images to line images.")
#     input_path = os.path.join(DATA_RAW, "ICFHR_2016", "page_image", "train", "Seite0001.JPG")
#     crop_image(input_path, "train")
#     logger.info("Cropping finished.")
