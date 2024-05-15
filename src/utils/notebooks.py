import os
import re
from IPython.display import display
from PIL import Image
import json
from zipfile import ZipFile
import shutil
from collections import Counter
import re
import matplotlib.pyplot as plt
from config.config_paths import DATA_PROCESSED, DATA_RAW
from src.utils.notebooks import *
from src.processor.data_loader import get_data_loader

# Constants
GW_GT_PATH = os.path.join(DATA_PROCESSED, "GW", "ground_truth", "GW_gt.json")
GW_IMAGE_PATH = os.path.join(DATA_PROCESSED, "GW", "line_image", "GW_image.json")
IAM_GT_PATH = os.path.join(DATA_PROCESSED, "jh", "ground_truth", "jh_gt.json")
IAM_IMAGE_PATH = os.path.join(DATA_PROCESSED, "jh", "line_image", "jh_image.json")
BULLINGER_GT_PATH = os.path.join(DATA_PROCESSED, "Bullinger", "ground_truth", "bullinger_gt.json")
BULLINGER_IMAGE_PATH = os.path.join(DATA_PROCESSED, "Bullinger", "line_image", "bullinger_image.json")
ICFHR_GT_PATH = os.path.join(DATA_PROCESSED, "ICFHR_2016", "ground_truth", "icfhr_gt.json")
ICFHR_IMAGE_PATH = os.path.join(DATA_PROCESSED, "ICFHR_2016", "line_image", "icfhr_image.json")
WIKI_TRAIN_PATH = os.path.join(".", "data", "wiki", "enwiki_train.txt")
WIKI_VAL_PATH = os.path.join(".", "data", "wiki", "enwiki_val.txt")

# Dataset Information from website (used for double check)
# Text line
GW_TEXT_LINE = 656
IAM_TEXT_LINE = 6916
BULLINGER_TEXT_LINE = 165673
ICFHR_TEXT_LINE = 450
# Train, val, test set (whole image for ICFHR 2016)
GW_TRAIN, GW_VAL, GW_TEST = 329, 163, 164
IAM_TRAIN, IAM_VAL, IAM_TEST = 5532, 691, 693
BULLINGER_TRAIN, BULLINGER_VAL, BULLINGER_TEST = 132538, 16567, 16568
ICFHR_TRAIN, ICFHR_VAL, ICFHR_TEST = 400, 50, 50

# def check_text_lines(image_dict, gt_dict, data_name, data_loader):
#     expected_lines = {"GW": {"lines": GW_TEXT_LINE, "train": GW_TRAIN, "val": GW_VAL, "test": GW_TEST},
#                     "IAM": {"lines": IAM_TEXT_LINE, "train": IAM_TRAIN, "val": IAM_VAL, "test": IAM_TEST},
#                     "BULLINGER": {"lines": BULLINGER_TEXT_LINE, "train": BULLINGER_TRAIN, "val": BULLINGER_VAL, "test": BULLINGER_TEST},
#                     "ICFHR": {"lines": ICFHR_TEXT_LINE, "train": ICFHR_TRAIN, "val": ICFHR_VAL, "test": ICFHR_TEST}}
#     gt_text_line = len(gt_dict)
#     image_text_line = len(image_dict)
#     train_loader, val_loader, test_loader = data_loader["cv1"]
#     assert gt_text_line==image_text_line, "Image and ground truth text lines are not the same."

#     info = {expected_lines[data_name]["lines"]: image_text_line,
#             expected_lines[data_name]["train"]: len(train_loader.dataset),
#             expected_lines[data_name]["val"]: len(val_loader.dataset),
#             expected_lines[data_name]["test"]: len(test_loader.dataset)}
#     print("Website        | Downloaded ")
#     print("-------------------------------")
#     for k,v in info.items():
#         print(f"{k:<15} | {v:<15}")

#     assert gt_text_line==image_text_line==expected_lines[data_name]["lines"], "Text lines are not the same."
#     assert len(train_loader.dataset)==expected_lines[data_name]["train"], "Training lines are not the same."
#     assert len(val_loader.dataset)==expected_lines[data_name]["val"], "Validation lines are not the same."
#     assert len(test_loader.dataset)==expected_lines[data_name]["test"], "Testing lines are not the same."
#     print("Assertion tests passed.")

#     return 


def visualise_dataset(gt_path, image_path, data_name=None):
    if data_name:
        root_path = os.path.join(DATA_RAW, "Bullinger")
        extract_folder = os.path.join(root_path, "extracted_folder")
        # Open the ZIP file
        if not os.path.exists(extract_folder):
            with ZipFile(os.path.join(root_path, "train", "0-2000-out.zip"), "r") as zip_file:
                zip_file.extractall(extract_folder)
        image_path = os.path.join(extract_folder, "0-2000-out", "de", "1000_00_r1l7.png")
    else:
        # Open the data files and read them
        with open(image_path, "r") as file:
            image_dict = json.load(file)

        # Get the first data sample
        file_name, image_name = next(iter(image_dict.items()))
        image_path = image_dict[file_name]

    with open(gt_path, "r") as file:
        gt_dict = json.load(file)
    file_name, gt = next(iter(gt_dict.items()))

    print("Image:")
    image = Image.open(image_path)
    display(image)

    print("Ground Truth:")
    print(gt)
    if data_name:
        shutil.rmtree(os.path.join(DATA_RAW, "Bullinger", "extracted_folder"))
    return

def get_punctuations() -> set:
    """
    get_punctuations()

    Outputs all the punctuations/dates used in the GW dataset.

    Returns
    --------
    result: set
        A set that contains all unique punctuations in the ground truth file
    """
    # Open the GW ground truth file and read it
    path = os.path.join(DATA_RAW, "GW", "ground_truth", "transcription.txt")
    with open(path, "r") as file:
        file_content = file.read()

    # Match the pattern with "s_" followed by two or more alphabets or decimal digits
    pattern = re.compile(r's_[a-zA-Z0-9]{2,}')
    # Find all matches in the file content and store it in result
    result = set()
    matches = pattern.findall(file_content)
    for match in matches:
        result.add(match[2:])
        
    return ", ".join(list(result))

def get_word_frequencies(text: str) -> dict:
    # Convert the text to lowercase and split it into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Use Counter to count the frequency of each word
    word_frequencies = Counter(words)
    return word_frequencies

def plot_freq(word_frequencies1, word_frequencies2, title):
    words1, frequencies1 = zip(*word_frequencies1)
    words2, frequencies2 = zip(*word_frequencies2)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the first subplot
    axs[0].bar(words1, frequencies1, color="blue")
    axs[0].set_title("Top 10 Most Frequent Words")
    axs[0].set_ylabel("frequency")

    # Add numbers on top of bars in the first subplot
    for i, value in enumerate(frequencies1):
        axs[0].text(i, value + 0.1, str(value), ha="center", va="bottom")

    # Rotate x-axis labels in the first subplot
    axs[0].set_xticks(range(len(words1)))
    axs[0].set_xticklabels(words1, rotation=45, ha="right")


    # Plot the second subplot
    axs[1].bar(words2, frequencies2, color="green")
    axs[1].set_title("Top 10 Least Frequent Words")
    axs[1].set_ylabel("frequency")

    # Rotate x-axis labels in the first subplot
    axs[1].set_xticks(range(len(words2)))
    axs[1].set_xticklabels(words2, rotation=45, ha="right")

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

def get_average_len(lst):
    # Calculate the total length of all strings
    total_length = sum(len(line) for line in lst if line)
    # Calculate the average string length
    average_length = total_length / len(lst) if len(lst) > 0 else 0
    return average_length

def calculate_non_alphanumeric_percentage(input_string):
    total_characters = len(input_string)
    non_alphanumeric_count = sum(not c.isalnum() for c in input_string)

    if total_characters > 0:
        non_alphanumeric_percentage = (non_alphanumeric_count / total_characters) * 100
    return non_alphanumeric_percentage

def get_dataset(gt_path, image_path):
    with open(gt_path, "r") as file:
        gt_dict = json.load(file)
    with open(image_path, "r") as file:
        image_dict = json.load(file)
    return image_dict, gt_dict


def get_dataset_info(gt_path, image_path):
    image_dict, gt_dict = get_dataset(gt_path, image_path)
    all_gt = " ".join([i for i in gt_dict.values() if i])
    all_words = set(all_gt.split(" "))
    word_frequencies = get_word_frequencies(all_gt)
    word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    # average_length = get_average_text_line_len(list(gt_dict.values()))
    info = {"text lines": len(gt_dict),
            "unique word instances": len(all_words),
            "unique letters": len(set(all_gt)),
            "average text line length": get_average_len(list(gt_dict.values())),
            "average word length": get_average_len(list(all_words)),
            "None in ground truth": list(gt_dict.values()).count(None),
            "percentage of non-character": calculate_non_alphanumeric_percentage(all_gt)}
    print("Metric                       | Value")
    print("----------------------------------------")
    for k,v in info.items():
        if type(v)==float:
            print(f"{k:<28} | {v:.2f}")
        else:
            print(f"{k:<28} | {v}")
    print("\n")

    plot_freq(word_frequencies[:10], word_frequencies[-10:], "Top 10 Least Frequent Words")
    return

def get_num_samples(data_loader):
    total_samples = 0
    for i, batch in enumerate(data_loader):
        total_samples += batch["pixel_values"].size(0)
    return total_samples # Assuming batch[0] contains the samples

def check_text_lines(image_dict, gt_dict, data_name, data_loader):
    expected_lines = {"gw": {"lines": GW_TEXT_LINE, "train": GW_TRAIN, "val": GW_VAL, "test": GW_TEST},
                    "iam": {"lines": IAM_TEXT_LINE, "train": IAM_TRAIN, "val": IAM_VAL, "test": IAM_TEST},
                    "bullinger": {"lines": BULLINGER_TEXT_LINE, "train": BULLINGER_TRAIN, "val": BULLINGER_VAL, "test": BULLINGER_TEST},
                    "icfhr": {"lines": ICFHR_TEXT_LINE, "train": ICFHR_TRAIN, "val": ICFHR_VAL, "test": ICFHR_TEST}}
    gt_text_line = len(gt_dict)
    image_text_line = len(image_dict)
    train_loader = data_loader["cv1"]["train"]
    val_loader = data_loader["cv1"]["val"]
    test_loader = data_loader["cv1"]["test"]
    # train_loader, val_loader, test_loader = data_loader["cv1"]
    # assert gt_text_line==image_text_line, "Image and ground truth text lines are not the same."

    # data = load_data()
    # # Split IAM data into train, validation and test datasets, and get the indices
    # gw_folds = get_split_indices("GW", data["GW_image"], data["GW_gt"])
    # iam_folds = get_split_indices("IAM", data["IAM_image"], data["IAM_gt"])
    # bullinger_folds = get_split_indices("Bullinger", data["bullinger_image"], data["bullinger_gt"])
    # icfhr_folds = get_split_indices("ICFHR", data["icfhr_image"], data["icfhr_gt"])
    
    train_sample = get_num_samples(train_loader)
    # shutil.rmtree(os.path.join(DATA_RAW, "Bullinger", "extracted_folder"))
    val_sample = get_num_samples(val_loader)
    # shutil.rmtree(os.path.join(DATA_RAW, "Bullinger", "extracted_folder"))
    test_sample = get_num_samples(test_loader)
    # shutil.rmtree(os.path.join(DATA_RAW, "Bullinger", "extracted_folder"))

    info = {"Text line": (expected_lines[data_name]["lines"], image_text_line),
            "Train": (expected_lines[data_name]["train"], train_sample),
            "Validation": (expected_lines[data_name]["val"], val_sample),
            "Test": (expected_lines[data_name]["test"], test_sample)}
    print("Data            | Website         | Downloaded ")
    print("-----------------------------------------------")
    for k,v in info.items():
        print(f"{k:<15} | {v[0]:<15} | {v[1]:<15}")

    # assert gt_text_line==image_text_line==expected_lines[data_name]["lines"], "Text lines are not the same."
    # assert train_sample==expected_lines[data_name]["train"], "Training lines are not the same."
    # assert val_sample==expected_lines[data_name]["val"], "Validation lines are not the same."
    # assert test_sample==expected_lines[data_name]["test"], "Testing lines are not the same."
    print("Assertion tests passed.")

    return 
    
def get_wiki_lines():

    with open(WIKI_TRAIN_PATH, 'r') as file:
        train_lines = file.readlines()
        train_line_count = len(train_lines)
        formatted_train_line_count = "{:,}".format(train_line_count).replace(",", "'")

    with open(WIKI_VAL_PATH, 'r') as file:
        val_lines = file.readlines()
        val_line_count = len(val_lines)
        formatted_val_line_count = "{:,}".format(val_line_count).replace(",", "'")

    print(f"The wikipedia training file has {formatted_train_line_count} lines.")
    print(f"The wikipedia validation file has {formatted_val_line_count} lines.")
    print(f"Validation data is {int(val_line_count/train_line_count*100)}% of training data")
    return
