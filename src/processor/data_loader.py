import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
from PIL import Image
from typing import List, Tuple
import os
from zipfile import ZipFile
import shutil
from src.utils.helpers import load_data
from config.config_paths import DATA_RAW
from transformers import VisionEncoderDecoderModel
import config.model_config as cfg

# Extracting the embedding layer from the model
# Load the processor and model from the specified TrOCR model
# processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(cfg.model_config["trocr_config"])

# Extracting the embedding layer from the model
embedding_layer = model.decoder.model.decoder.embed_tokens

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
    def __init__(self, data: dict, labels: dict, processor, max_target_length, embedding_layer=embedding_layer, transform=None, data_name=None):
        self.data = data
        self.labels = labels
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform
        self.data_name = data_name
        self.trocr_embedding = embedding_layer

    def shift_tokens_right(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last non pad token."""
        # Convert input_ids list to a tensor
        input_ids = torch.tensor(input_ids)
        # Ensure input_ids is 2D: [1, sequence_length]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        # Clone the input_ids to create prev_output_tokens tensor
        prev_output_tokens = input_ids.clone()
        # Calculate the index of the end of sequence (EOS) token for each sequence in the batch
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        # Shift the tokens to the right
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        
        return prev_output_tokens

    def create_attention_mask(self, input_ids, pad_token_id):
        # Create a mask of the same shape as input_ids, where each element is 1 if the corresponding input_id is not
        # the pad_token_id, and 0 if it is the pad_token_id.
        return (input_ids != pad_token_id).long().squeeze()

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
        if self.data_name:
            root_path = os.path.join(DATA_RAW, "Bullinger")
            extract_folder = os.path.join(root_path, "extracted_folder")
            # Open the ZIP file
            mode = index.split("_")[0]
            if not os.path.exists(extract_folder):
                for zip_filename in os.listdir(os.path.join(root_path, mode)):
                    if zip_filename.endswith(".zip"):
                        with ZipFile(os.path.join(root_path, mode, zip_filename), "r") as zip_file:
                            zip_file.extractall(extract_folder)
            sample_path = os.path.join(extract_folder, sample_path)

        with open(sample_path, "rb") as file:
            sample = Image.open(file).convert("RGB")
            pixel_values = self.processor(sample, return_tensors="pt").pixel_values

            # if self.transform:
            #     sample = self.transform(sample)
        decoder_input_ids = self.processor.tokenizer(self.labels[index], 
                                        padding="max_length", 
                                        max_length=self.max_target_length).input_ids
        

        decoder_input_ids = self.shift_tokens_right(decoder_input_ids, self.processor.tokenizer.pad_token_id)
        # Get attention mask
        attention_mask = self.create_attention_mask(decoder_input_ids, self.processor.tokenizer.pad_token_id)

        # print("decoder input ids type: ", decoder_input_ids.dtype)
        # print("decoder input ids: ", decoder_input_ids.size())

        # Get embeddings
        labels = self.trocr_embedding(decoder_input_ids)
        # print("labels ###: ", labels.size())
        if labels.shape[0] == 1:
            labels = labels.squeeze(0)
        # labels = labels.squeeze(1)
        # label = self.labels[index]
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values.squeeze(), 
                "label_emb": labels, 
                "label_str": self.labels[index], 
                "labels": decoder_input_ids, 
                "attention_mask": attention_mask}

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
    def get_indices_gw():
        base_path = os.path.join(DATA_RAW, "GW", "cv")
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
    def get_indices_iam():
        fold = tuple()
        base_path = os.path.join(DATA_RAW, "IAM", "split")
        # Open the file in read mode
        for filename in ["trainset.txt", "validationset1.txt", "testset.txt"]:
            with open(os.path.join(base_path, filename), "r") as file:
                lines = file.readlines()
            lines = [i.rstrip("\n") for i in lines]
            fold = fold + (lines,)
        folds = [fold]
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
    functions = {"GW": get_indices_gw, "IAM": get_indices_iam, "Bullinger": get_indices_bullinger, "ICFHR": get_indices_icfhr}
    return functions.get(data_name, lambda: "Invalid data_name")()

def process_data_loader(image: dict, gt: dict, folds: list, batch_size: int, processor, max_target_length, transform=None, data_name=None) -> dict:
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
    custom_dataset = CustomDataset(data=image, labels=gt, processor=processor, max_target_length=max_target_length, transform=transform, data_name=data_name)
    # Create instances of my custom dataset for training, validation and testing purposes
    for index, fold in enumerate(folds):
        # Create DataLoader for training, validation, and test sets
        train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, sampler=torch.utils.data.SubsetRandomSampler(fold[0]))
        val_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, sampler=torch.utils.data.SubsetRandomSampler(fold[1]))
        test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, sampler=torch.utils.data.SubsetRandomSampler(fold[2]))
        data_loaders["cv"+str(index+1)] = {"train": train_loader, "val": val_loader, "test": test_loader}
    return data_loaders

def get_data_loader(batch_size: int, processor, max_target_length) -> tuple:
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

    # Load the data loaders for all datasets
    gw_data_loaders = process_data_loader(data["GW_image"], data["GW_gt"], GW_folds, batch_size, processor, max_target_length)
    iam_data_loaders = process_data_loader(data["IAM_image"], data["IAM_gt"], IAM_folds, batch_size, processor, max_target_length)
    bullinger_data_loaders = process_data_loader(data["bullinger_image"], data["bullinger_gt"], bullinger_folds, batch_size, processor, max_target_length, "Bullinger")
    icfhr_data_loaders = process_data_loader(data["icfhr_image"], data["icfhr_gt"], icfhr_folds, batch_size, processor, max_target_length)
    return gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders

if __name__=="__main__":
    gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(512)
    test_loader = iam_data_loaders["cv1"]["train"]
    for batch in test_loader:
        # Extract the image tensor from the batch (adjust based on your actual data structure)
        images = batch[0]
        # Check if images is a tensor and has the expected shape (3 channels for RGB images)
        if isinstance(images, torch.Tensor):
            print("Images are present in the batch.")
        else:
            print("No images found in the batch.")
    # shutil.rmtree(os.path.join(".", "data", "raw", "Bullinger", "extracted_folder"))
    # train_en_loader, val_en_loader, test_en_loader = get_data_loader(512, 0.2)
    # # Iterate through the DataLoader
    # for batch_idx, (data, labels) in enumerate(train_en_loader):
    #     "data" contains input data, and "labels" contains corresponding labels
    #     print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {len(labels)}")