import os
import json
import configparser
from typing import List, Tuple
from config.config_paths import DATA_PROCESSED, CONFIG_CONST
import torch
import tensorflow as tf
import torch.nn as nn
import random
import re
from langdetect import detect, LangDetectException
import wikipedia
from tqdm import tqdm

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

class DummyLayer(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy layer. It should not be called.")
    
def write_predictions_to_file(pred_str, label_str, file_path):
    with open(file_path, "a") as f:
        for pred, label in zip(pred_str, label_str):
            f.write("output: " + pred + "\n")
            f.write("label:  " + label + "\n\n")
    return

def shutdown_logger(logger):
    """
    Shuts down the specified logger.

    Args:
        logger (logging.Logger): The logger to shut down.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def set_gpu(gpu: str):
    # Specify which GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  # for example, GPU 1 and 2

    # List all visible GPUs after setting the environment variable
    physical_devices = tf.config.list_physical_devices("GPU")

    # Enable memory growth for each visible GPU
    for gpu in physical_devices:
        try:
            # Set memory growth to True
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for {gpu}")
        except RuntimeError as e:
            # Print any RuntimeErrors and continue
            print(e)
    return

def is_english(text):
    try:
        # Use langdetect to detect the language
        language = detect(text)
        # Check if the detected language is English
        return language == 'en'
    except LangDetectException:
        # In case the language detection fails or the text is too short
        return False

def wiki_dataset(sample_size=100000, sample_articles=5, sample_sentences=2):
    file_path = "./data/wiki/enwiki_title.txt"

    with open(file_path, 'r') as file:
        lines = list(file)
    
    # If the file has fewer lines than the sample size, return all lines
    if len(lines) < sample_size:
        return lines
    
    # Otherwise, randomly sample 'sample_size' lines from the list
    articles = random.sample(lines, sample_size)
    
    # Optionally, strip newline characters from sampled lines
    articles = [line.strip() for line in articles]
    final_sentences = []
    for article in tqdm(articles):
        page_ids = wikipedia.search(article)
        page_ids = random.sample(page_ids, sample_articles) if len(page_ids) >= sample_articles else page_ids
        for id in page_ids:
            try:
                page = wikipedia.page(id)
                content = page.content
                content = content.replace("\n", "")
                content = re.sub(r'=+.*?=+', ' ', content)
                # sentences = re.split(r'\. |\.\n', content)
                sentences = content.split(". ")
                # Sample 2 sentences randomly from the list of sentences
                sampled_sentences = random.sample(sentences, sample_sentences) if len(sentences) >= sample_sentences else sentences
                final_sentences += [line+". ".strip() for line in sampled_sentences if is_english(line)]
            except:
                pass
    print(final_sentences)
    with open("./data/wiki/enwiki_sentence.txt", 'w') as file:
        # Iterate over each string in the list
        for string in final_sentences:
            # Write the string followed by a newline character to the file
            file.write(f"{string}\n")
    return
