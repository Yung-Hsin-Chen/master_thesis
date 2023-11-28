from PIL import Image
from config.config import configure_logging
import logging
import os
import xml.etree.ElementTree as ET

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

def get_coordinate(input_path) -> tuple:
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
#     logger.info("Start cropping ICFHR 2016 images to line images...")
#     input_path = os.path.join(".", "data", "raw", "ICFHR_2016", "page_image", "train", "Seite0001.JPG")
#     # coordinates = "1828,422 1617,420 285,415 285,635 1617,640 1828,642"
#     crop_image(input_path, "train")
