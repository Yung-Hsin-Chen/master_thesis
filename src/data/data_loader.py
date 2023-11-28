import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import yaml
from PIL import Image
from typing import List, Tuple
import os
import json
from zipfile import ZipFile
import shutil

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
    def __init__(self, data: dict, labels: dict, transform=None, data_name=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.data_name = data_name

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
        sample_path = self.data[index]
        # print(sample_path)
        if self.data_name:
            root_path = os.path.join(".", "data", "raw", "Bullinger")
            extract_folder = os.path.join(root_path, "extracted_folder")
            # Open the ZIP file
            mode = index.split("_")[0]
            if not os.path.exists(extract_folder):
                for zip_filename in os.listdir(os.path.join(root_path, mode)):
                    if zip_filename.endswith(".zip"):
                        with ZipFile(os.path.join(root_path, mode, zip_filename), "r") as zip_file:
                            zip_file.extractall(extract_folder)
            sample_path = os.path.join(extract_folder, sample_path)

        # else:
        with open(sample_path, "rb") as file:
            sample = Image.open(file)
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
    file_path = os.path.join(".", "data", "processed", "Bullinger", "ground_truth", "bullinger_gt.json")
    with open(file_path, "r") as json_file:
        bullinger_gt = json.load(json_file)
    file_path = os.path.join(".", "data", "processed", "ICFHR_2016", "ground_truth", "icfhr_gt.json")
    with open(file_path, "r") as json_file:
        icfhr_gt = json.load(json_file)
    # Load image data
    file_path = os.path.join(".", "data", "processed", "GW", "line_image", "GW_image.json")
    with open(file_path, "r") as json_file:
        GW_image = json.load(json_file)
    file_path = os.path.join(".", "data", "processed", "IAM", "line_image", "IAM_image.json")
    with open(file_path, "r") as json_file:
        IAM_image = json.load(json_file)
    file_path = os.path.join(".", "data", "processed", "Bullinger", "line_image", "bullinger_image.json")
    with open(file_path, "r") as json_file:
        bullinger_image = json.load(json_file)
    file_path = os.path.join(".", "data", "processed", "ICFHR_2016", "line_image", "icfhr_image.json")
    with open(file_path, "r") as json_file:
        icfhr_image = json.load(json_file)
    return {"GW_image": GW_image, "GW_gt": GW_gt,
            "IAM_image": IAM_image, "IAM_gt": IAM_gt,
            "bullinger_image": bullinger_image, "bullinger_gt": bullinger_gt,
            "icfhr_image": icfhr_image, "icfhr_gt": icfhr_gt}

def resize_data() -> tuple:
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

def get_split_indices(data_name: str, image: dict, gt: dict) -> List[Tuple[List, List, List]]:
    """
    Get indices for splitting the dataset into training, validation, and test sets based on the specified data_name.

    Parameters
    -----------
    data_name: str
        Name of the dataset, e.g., "GW", "IAM", "Bullinger", "ICFHR".
    image: dict
        Dictionary containing image data.
    gt: dict
        Dictionary containing ground truth data.

    Returns
    --------
    List[Tuple[List, List, List]]
        A list of tuples, where each tuple represents the indices for training, validation, and test sets.
        The order of tuples corresponds to the number of folds (e.g., for cross-validation).
    
    - ValueError: If an unsupported data_name is provided.

    Notes
    ------
    - For "GW" (George Washington) dataset, indices are read from predefined files in the "data/raw/GW/cv" directory.
    - For "IAM" dataset, indices are split into training, validation, and test sets using train_test_split function.
    - For "Bullinger" dataset, indices are separated based on prefixes ("train", "val", "test").
    - For "ICFHR" dataset, indices are separated based on prefixes ("train", "val", "test").
    """
    folds = []
    # Get indices for GW
    def get_indices_GW():
        base_path = os.path.join(".", "data", "raw", "GW", "cv")
        # Open the file in read mode
        for cv_dir in ["cv1", "cv2", "cv3", "cv4"]:
            one_fold = tuple()
            for cv_file in ["train.txt", "valid.txt", "test.txt"]:
                with open(os.path.join(base_path, cv_dir, cv_file), "r") as file:
                    lines = file.readlines()
                lines = [i.rstrip("\n") for i in lines]
                one_fold = one_fold + (lines,)
            folds.append(one_fold)
        return folds
    # Get indices for IAM
    def get_indices_IAM():
        indices_list = list(image.keys())
        # Split the indices into training, validation and testing sets for each fold
        num_folds = 1
        for fold in range(num_folds):
            train_indices, test_indices = train_test_split(indices_list, test_size=0.25, random_state=42)
            train_indices, val_indices = train_test_split(train_indices, test_size=1/3, random_state=42)
            folds.append((train_indices, val_indices, test_indices))
        return folds
    # Get indices for Bullinger
    def get_indices_bullinger():
        indices_list = list(image.keys())
        train_indices = [index for index in indices_list if index.split("_")[0]=="train"]
        val_indices = [index for index in indices_list if index.split("_")[0]=="val"]
        test_indices = [index for index in indices_list if index.split("_")[0]=="test"]
        folds = [(train_indices, val_indices, test_indices)]
        return folds
    # Get indices for ICFHR
    def get_indices_icfhr():
        indices_list = list(image.keys())
        train_indices = [index for index in indices_list if index.split("__")[0]=="train"]
        val_indices = [index for index in indices_list if index.split("__")[0]=="val"]
        test_indices = [index for index in indices_list if index.split("__")[0]=="test"]
        folds = [(train_indices, val_indices, test_indices)]
        return folds
    # A dictionary to map data_name to the corresponding function
    functions = {"GW": get_indices_GW, "IAM": get_indices_IAM, "Bullinger": get_indices_bullinger, "ICFHR": get_indices_icfhr}
    return functions.get(data_name, lambda: "Invalid data_name")()

def process_data_loader(image: dict, gt: dict, folds: list, batch_size: int, transform, data_name=None) -> dict:
    """
    Process and create data loaders with/without a cross-validation setup.

    Parameters
    -----------
    image: dict 
        Dictionary containing image paths.
    gt: dict
        Dictionary containing ground truth text data.
    folds: list
        List of tuples representing indices for training, validation, and test sets.
    batch_size: int
    transform
        Data transformation to be applied to the samples.
    data_name :Optional[str]
        Name of the dataset (default is None).

    Returns
    --------
    dict
        A dictionary containing data loaders for each cross-validation split.
        The keys are in the form "cv1", "cv2", ..., "cvN".
        The values are tuples of DataLoader objects for training, validation, and test sets.
    """
    data_loaders = dict()
    custom_dataset = CustomDataset(data=image, labels=gt, transform=transform, data_name=data_name)
    # Create instances of my custom dataset for training, validation and testing purposes
    for index, fold in enumerate(folds):
        # Create DataLoader for training, validation, and test sets
        train_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(fold[0]))
        val_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(fold[1]))
        test_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(fold[2]))
        data_loaders["cv"+str(index+1)] = (train_loader, val_loader, test_loader)
    return data_loaders

def get_data_loader(batch_size: int) -> tuple:
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
    # Assert data types and value range 
    assert isinstance(batch_size, int)
    # Load and prepare data for splitting
    data = load_data()
    # Split IAM data into train, validation and test datasets, and get the indices
    GW_folds = get_split_indices("GW", data["GW_image"], data["GW_gt"])
    IAM_folds = get_split_indices("IAM", data["IAM_image"], data["IAM_gt"])
    bullinger_folds = get_split_indices("Bullinger", data["bullinger_image"], data["bullinger_gt"])
    icfhr_folds = get_split_indices("ICFHR", data["icfhr_image"], data["icfhr_gt"])
    # Define data transformations
    resize_height, resize_width = resize_data()
    # Assert data types 
    assert isinstance(resize_height, int)
    assert isinstance(resize_width, int)
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor()
    ])
    # Load the data loaders for all datasets
    GW_data_loaders = process_data_loader(data["GW_image"], data["GW_gt"], GW_folds, batch_size, transform)
    IAM_data_loaders = process_data_loader(data["IAM_image"], data["IAM_gt"], IAM_folds, batch_size, transform)
    bullinger_data_loaders = process_data_loader(data["bullinger_image"], data["bullinger_gt"], bullinger_folds, batch_size, transform, "Bullinger")
    icfhr_data_loaders = process_data_loader(data["icfhr_image"], data["icfhr_gt"], icfhr_folds, batch_size, transform)
    return GW_data_loaders, IAM_data_loaders, bullinger_data_loaders, icfhr_data_loaders

if __name__=="__main__":
    GW_data_loaders, IAM_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(512)
    test_loader = icfhr_data_loaders["cv1"][0]
    for batch in test_loader:
        # Extract the image tensor from the batch (adjust based on your actual data structure)
        images = batch[0]
        # Check if images is a tensor and has the expected shape (3 channels for RGB images)
        if isinstance(images, torch.Tensor):
            print("Images are present in the batch.")
        else:
            print("No images found in the batch.")g
    # shutil.rmtree(os.path.join(".", "data", "raw", "Bullinger", "extracted_folder"))
    # train_en_loader, val_en_loader, test_en_loader = get_data_loader(512, 0.2)
    # # Iterate through the DataLoader
    # for batch_idx, (data, labels) in enumerate(train_en_loader):
    #     "data" contains input data, and "labels" contains corresponding labels
    #     print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {len(labels)}")