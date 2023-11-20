import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import yaml
from PIL import Image
from typing import List
import os
import json

# Define a custom dataset class 
class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling a collection of data samples with corresponding labels.

    Args
    -----
    data: list
        A list of file paths or data samples.
    labels: list
        A list of corresponding labels.
    transform: Optional[callable])
        A callable object (e.g., torchvision.transforms.Compose) that applies transformations to the data samples. Default is None.

    Attributes
    -----------
    data: list
        A list of file paths or data samples.
    labels: list
        A list of corresponding labels.
    transform: Optional[callable])
        A callable object that applies transformations to the data samples.

    Methods
    --------
    __len__(): Returns the number of samples in the dataset.
    __getitem__(index: int) -> Tuple: Retrieves a data sample and its corresponding label at the specified index.

    Note
    ------
    Ensure that the PIL library is installed (`pip install Pillow`) for working with images.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        --------
        len(self.data): int
            The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a data sample and its corresponding label at the specified index.

        Parameters
        -----------
        index: int
            The index of the sample to retrieve.

        Returns
        --------
        sample, label: Tuple
            A tuple containing the data sample and its corresponding label.
        """
        sample = self.data[index]
        sample = Image.open(sample)

        if self.transform:
            sample = self.transform(sample)

        label = self.labels[index]

        return sample, label

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
    file_path = os.path.join(".", "data", "processed", "GW", "ground_truth", "GW_gt.json")
    with open(file_path, "r") as json_file:
        GW_gt = json.load(json_file)
    file_path = os.path.join(".", "data", "processed", "IAM", "ground_truth", "IAM_gt.json")
    with open(file_path, "r") as json_file:
        IAM_gt = json.load(json_file)
    # Load image data
    file_path = os.path.join(".", "data", "processed", "GW", "line_image", "GW_image.json")
    with open(file_path, "r") as json_file:
        GW_image = json.load(json_file)
    file_path = os.path.join(".", "data", "processed", "IAM", "line_image", "IAM_image.json")
    with open(file_path, "r") as json_file:
        IAM_image = json.load(json_file)
    # TODO load German data
    GW_image.update(IAM_image)
    GW_gt.update(IAM_gt)
    return {"en_image": GW_image, "en_gt": GW_gt}

def align_image_gt(image: dict, gt: dict) -> dict:
    """
    Align images and their respective ground truth by the file name.

    Parameters
    -----------
    image: dict
    gt: dict
        The file names are keys and the data contents as values.

    Returns
    --------
    {"en_image": image_list, "en_gt": gt_list}: dict
        The keys are the name of the dataset, and the values are the data in lists.
        The file names of each sample are taken out.
    """
    file_names = list(gt.keys())
    image_list = [image[file_name] for file_name in file_names]
    gt_list = [gt[file_name] for file_name in file_names]
    return {"en_image": image_list, "en_gt": gt_list}

def get_resize_dim() -> tuple:
    """
    Get the resize height and the resize weight from config_const.yaml.

    Returns
    --------
    resize_data["resize_height"], resize_data["resize_width"]: tuple
    """
    # Load resize dimensions from the YAML file
    with open(os.path.join(".", "config", "config_const.yaml"), "r") as config_file:
        resize_data = yaml.safe_load(config_file)["training"]
    return resize_data["resize_height"], resize_data["resize_width"]

def get_data_loader(batch_size: int, test_size: float) -> List[DataLoader]:
    """
    Loads and prepares data for training, validation, and testing, and returns corresponding DataLoaders.

    Parameters
    -----------
    batch_size: int
        The batch size for the DataLoaders.
    test_size: float
        The proportion of the dataset to include in the test split (0.0 to 1.0).

    Returns
    --------
    List[DataLoader]
        A list containing DataLoader instances for training, validation, and testing.

    """
    # Load and prepare data for splitting
    data = load_data()
    en_data = align_image_gt(data["en_image"], data["en_gt"])
    # Split data into train, validation and test datasets
    train_en_image, test_en_image, train_en_gt, test_en_gt = train_test_split(en_data["en_image"], en_data["en_gt"], test_size=test_size, random_state=42)
    train_en_image, val_en_image, train_en_gt, val_en_gt = train_test_split(train_en_image, train_en_gt, test_size=test_size, random_state=42)
    # Define data transformations
    resize_height, resize_width = get_resize_dim()
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor()
    ])
    # Create instances of my custom dataset for training, validation and testing purposes
    train_en_dataset = CustomDataset(data=train_en_image, labels=train_en_gt, transform=transform)
    val_en_dataset = CustomDataset(data=val_en_image, labels=val_en_gt, transform=transform)
    test_en_dataset = CustomDataset(data=test_en_image, labels=test_en_gt, transform=transform)
    # Create a DataLoader
    train_en_loader = DataLoader(dataset=train_en_dataset, batch_size=batch_size, shuffle=True)
    val_en_loader = DataLoader(dataset=val_en_dataset, batch_size=batch_size, shuffle=True)
    test_en_loader = DataLoader(dataset=test_en_dataset, batch_size=batch_size, shuffle=True)
    return [train_en_loader, val_en_loader, test_en_loader]

if __name__=="__main__":
    train_en_loader, val_en_loader, test_en_loader = get_data_loader(512, 0.2)
    # Iterate through the DataLoader
    for batch_idx, (data, labels) in enumerate(train_en_loader):
        # "data" contains input data, and "labels" contains corresponding labels
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {len(labels)}")