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
from config.config_paths import DATA_RAW, DATA_PROCESSED
from transformers import VisionEncoderDecoderModel
import config.model_config as cfg
import torch.nn.functional as F
import collections
import copy
import random
import nltk
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import json
nltk.download('punkt')

# Extracting the embedding layer from the model
# Load the processor and model from the specified TrOCR model
# processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(cfg.model_config["trocr_config"])

# Extracting the embedding layer from the model
embedding_layer = model.decoder.model.decoder.embed_tokens

charbert_dataset_args = cfg.charbert_dataset_args

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
    
    def get_label_tensor(self, decoder_input_ids, pad_token_id):
        decoder_input_ids = [label if label != pad_token_id else -100 for label in decoder_input_ids]
        decoder_input_ids = decoder_input_ids[1:] + [-100]
        decoder_input_ids = torch.tensor(decoder_input_ids)
        # One-hot encode
        # vocab_size = 50265  # The vocabulary size of the tokenizer
        # one_hot_encoded = F.one_hot(decoder_input_ids, num_classes=vocab_size).float()
        # return one_hot_encoded
        return decoder_input_ids


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
        # print(self.data)
        # print(index)
        sample_path = self.data[index]
        # sample_path = sample_path.replace(".", "/home/user/yunghsin/master_thesis/master_thesis", 1)
        # print(sample_path)
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
            input_size = (384, 384)
            norm_tfm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            resize_tfm = transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC)

            # Compose the transforms
            transform = transforms.Compose([
                resize_tfm, 
                transforms.ToTensor(),
                norm_tfm
            ])
            sample = Image.open(file).convert("RGB")
            pixel_values = transform(sample)
            # not_ones = pixel_values[pixel_values != 1]

            # if self.transform:
            #     sample = self.transform(sample)
        # decoder_input_ids = self.processor.tokenizer(self.labels[index], 
        #                                 padding="max_length", 
        #                                 max_length=self.max_target_length).input_ids
        decoder_input_ids = self.processor.tokenizer(self.labels[index], padding="max_length", max_length=512).input_ids

        label_tensor = self.get_label_tensor(decoder_input_ids, self.processor.tokenizer.pad_token_id)
        decoder_input_ids = torch.tensor(decoder_input_ids)
        # decoder_input_ids = self.shift_tokens_right(decoder_input_ids, self.processor.tokenizer.pad_token_id)
        # Get attention mask
        attention_mask = self.create_attention_mask(decoder_input_ids, self.processor.tokenizer.pad_token_id)
        # charbert_attention_mask = self.create_charbert_attention_mask(decoder_input_ids, self.processor.tokenizer.pad_token_id)

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
        # print(self.labels[index])
            # print(self.labels[index])
        # print(decoder_input_ids)
        # print("data_loader: ", labels)
        # print("In CustomDataset: ", self.labels[index])
        return {"pixel_values": pixel_values.squeeze(), 
                "label_emb": labels, 
                "label_str": self.labels[index], 
                "labels": label_tensor, 
                "attention_mask": attention_mask,
                "test_decoder_input_ids": decoder_input_ids}
    
class CharBertDataset(Dataset):
    def __init__(self, args, data: dict, labels: dict, block_size=512):
        self.data = data
        self.labels = labels
        # assert os.path.isfile(file_path)
        self.char2ids_dict = self.load_line_to_ids_dict(fname=args.char_vocab)
        self.term2ids_dict = self.load_line_to_ids_dict(fname=args.term_vocab)
        # self.model = initialise_charbert_model(experiment_version=None)
        # directory, filename = os.path.split(file_path)
        # cached_features_file = os.path.join(directory, '_cached_lm_' + str(block_size) + '_' + filename)

        # file_raws = 0
        # with open(file_path, 'r', encoding="utf-8") as f:
        #     for _ in f:
        #         file_raws += 1
        # self.file_raws = file_raws
        # self.nraws = args.input_nraws
        self.shuffle = True
        # self.file_path = file_path
        # self.finput = open(file_path, encoding="utf-8")
        self.current_sample_idx = -1
        self.examples = []
        self.tokenizer = args.tokenizer
        self.block_size = args.block_size
        self.num_nraws = 0
        self.args = args
        self.rng = random.Random(args.seed)
        # self.tokenizer = tokenizer

    def get_charbert_label(self, char_input_ids, start_ids, end_ids, input_ids):
        outputs = self.model(char_input_ids=char_input_ids, start_ids=start_ids, end_ids=end_ids, input_ids=input_ids)
        return outputs

    def create_input_tokens(self, tokens):
        vocab = self.tokenizer.get_vocab()

        # Now, `vocab` is a dictionary where keys are tokens and values are token IDs
        vocab_words = list(vocab.keys())
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        self.rng.shuffle(cand_indexes)

        output_tokens = list(tokens)
        return output_tokens

    def read_nraws(self, text):
        # self.num_nraws += 1
        # logger.info("Reading the [%d]th data block from dataset file at %s" % (self.num_nraws, self.file_path))

        # text = ""
        # for _ in range(self.nraws):
        #     line =  self.finput.readline()
        #     if line:
        #         text += line.strip()
        #     else:
        #         self.finput.seek(0)
        #         line =  self.finput.readline()
        #         text += line.strip()

        # line = self.finput.readline()
        # if not line:
        #     self.finput.seek(0)
        #     line = self.finput.readline()

        # text = line.strip()
        # print("\ntext: ", text)
                
        doc_tokens = nltk.word_tokenize(text)
        # print("\ndoc_tokens: ", doc_tokens)
        if self.args.output_debug:
            print(f"doc_tokens : {' '.join(doc_tokens)}")

        tokenized_tokens = []
        sub_index_to_orig_token = {}
        sub_index_to_change = {}
        num_diff = num_same = 0
        for idx, token in enumerate(doc_tokens):
            ori_token = copy.deepcopy(token)
            if ori_token != token and self.args.output_debug:
                if num_diff % 1000 == 0:
                    print(f"Change the token {ori_token} To {token}")
                num_diff += 1
            else:
                num_same += 1
            sub_tokens = []
            if self.args.model_type == 'roberta':
                sub_tokens = self.tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = self.tokenizer.tokenize(token)
            for sub_w in sub_tokens:
                sub_index_to_orig_token[len(tokenized_tokens)] = token
                if ori_token != token:
                    sub_index_to_change[len(tokenized_tokens)] = True
                else:
                    sub_index_to_change[len(tokenized_tokens)] = False
                    
                tokenized_tokens.append(sub_w)
        if self.args.output_debug:
            print(f"num_same: {num_same} num_diff: {num_diff}")
            print(f"tokenized doc: {' '.join(tokenized_tokens)}")

        # input_tokens, mask_labels = self.create_masked_lm_predictions(tokenized_tokens,\
        #         self.args.mlm_probability, self.tokenizer, self.rng, sub_index_to_change)
        input_tokens = self.create_input_tokens(tokenized_tokens)
        # print("\ninput_tokens: ", input_tokens)
        tokenized_text = self.tokenizer.convert_tokens_to_ids(input_tokens)
        # print("\ntokenized_text: ", tokenized_text)

        # seq_maxlen = self.block_size - 2
        # self.examples = []
        # for i in range(0, len(tokenized_text)-seq_maxlen+1, seq_maxlen): # Truncate in block of block_size
        #     print("in the loop")
        #     input_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+seq_maxlen])
        #     labels = [-1] + input_ids + [-1] #For CLS and SEP
        #     char_input_ids, start_ids, end_ids = self.build_char_inputs(input_ids, sub_index_to_orig_token, i, self.rng, labels)
        #     assert len(input_ids) == len(labels)
        #     assert len(input_ids) == len(start_ids)
        #     assert len(input_ids) == len(end_ids)
        #     self.examples.append((torch.tensor(char_input_ids), torch.tensor(start_ids), torch.tensor(end_ids),\
        #         torch.tensor(input_ids), torch.tensor(labels)))
        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
        # labels = [-1] + input_ids + [-1] #For CLS and SEP
        char_input_ids, start_ids, end_ids = self.build_char_inputs(input_ids, sub_index_to_orig_token, 0, self.rng)
        # assert len(input_ids) == len(labels)
        # print("block size: ", self.block_size)
        # print(self.tokenizer.max_len_single_sentence)
        attention_mask = [1]*len(input_ids)
        input_ids = input_ids + [1]*(self.block_size-len(input_ids))
        attention_mask = attention_mask + [0]*(self.block_size-len(attention_mask))
        # print("\nlen input_ids: ", len(input_ids))
        # print("\nlen start_ids: ", len(start_ids))
        # print("\nlen attention_mask: ", len(attention_mask))
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) == len(start_ids)
        assert len(input_ids) == len(end_ids)
        # print("\nchar_input_ids: ", char_input_ids[:15])
        # print("\nstart_ids: ", start_ids[:15])
        # print("\nend_ids: ", end_ids[:15])
        # print("\ninput_ids: ", input_ids[:15])
        # label = self.get_charbert_label(char_input_ids=torch.tensor(char_input_ids), start_ids=torch.tensor(start_ids), end_ids=torch.tensor(end_ids), input_ids=torch.tensor(input_ids))
        self.examples = {"char_input_ids": torch.tensor(char_input_ids), "start_ids": torch.tensor(start_ids),
                        "end_ids": torch.tensor(end_ids), "input_ids": torch.tensor(input_ids),
                        "charbert_attention_mask": torch.tensor(attention_mask)}
                        # "charbert_token_labels": label[0], "charbert_char_labels": label[2]}
        # self.examples.append(data)
        # self.examples.append((torch.tensor(char_input_ids), torch.tensor(start_ids), torch.tensor(end_ids),\
        #     torch.tensor(input_ids)))
        # print("example: ", self.examples)
        # self.current_sample_idx = -1
        # if self.shuffle:
        #     random.shuffle(self.examples)
        return

    def build_char_inputs(self, input_ids, sub_index_to_ori_tok, start, rng):
        all_seq_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        #if True:
        if self.args.output_debug:
            print(f"all_seq_tokens: {' '.join(all_seq_tokens)}")
        char_ids = []
        start_ids = []
        end_ids = []
        char_maxlen = self.args.block_size * self.args.char_maxlen_for_word
        for idx, token in enumerate(all_seq_tokens):
            if len(char_ids) >= char_maxlen:
                break
            token = token.strip("##")
            if token == self.tokenizer.unk_token:
                tok_orig_index = idx+start - 1
                if tok_orig_index in sub_index_to_ori_tok : #-1 for CLS
                    orig_token = sub_index_to_ori_tok[tok_orig_index]
                    #print(f'UNK: {token} to orig_tokens: {orig_token}')    
                    token = orig_token
            if token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]:# or labels[idx] != -1:
                start_ids.append(len(char_ids))
                end_ids.append(len(char_ids))
                char_ids.append(0)
            else:
                for char_idx, c in enumerate(token):
                    if len(char_ids) >= char_maxlen:
                        break
                    
                    if char_idx == 0:
                        start_ids.append(len(char_ids))
                    if char_idx == len(token) - 1:
                        end_ids.append(len(char_ids))

                    if c in self.char2ids_dict:
                        cid = self.char2ids_dict[c]
                    else:
                        cid = self.char2ids_dict["<unk>"]
                    char_ids.append(cid)

            if len(char_ids) < char_maxlen:
                char_ids.append(0)
            #if True:
            if self.args.output_debug:
                print(f'token[{token}]: {" ".join(map(str, char_ids[-1*(len(token)+2):]))}')
        #print(f'len of char_ids: {len(char_ids)}')
        if len(char_ids) > char_maxlen:
            char_ids = char_ids[:char_maxlen]
        else:
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
        while len(start_ids) < self.args.block_size:
            start_ids.append(char_maxlen-1)
        while len(end_ids) < self.args.block_size:
            end_ids.append(char_maxlen-1)
        #if True:
        if self.args.output_debug:
            print(f'char_ids : {" ".join(map(str, char_ids))}')
            print(f'start_ids: {" ".join(map(str, start_ids))}')
            print(f'end_ids  : {" ".join(map(str, end_ids))}')
        #max_start = max(start_ids)
        #max_end   = max(end_ids)
        #if max_start > char_maxlen or max_end > char_maxlen:
        #    print("Error sequence information")
        #    exit(0)
        return char_ids, start_ids, end_ids

    def load_line_to_ids_dict(self, fname):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(fname, "r", encoding="utf-8") as reader:
            chars = reader.readlines()
        for index, char in enumerate(chars):
            char = char.rstrip('\n')
            vocab[char] = index
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # char_ids, start_ids, end_ids = self.build_char_inputs(input_ids, sub_index_to_orig_token, i, self.rng, labels)
        # self.examples.append((torch.tensor(char_input_ids), torch.tensor(start_ids), torch.tensor(end_ids),\
        #         torch.tensor(input_ids), torch.tensor(labels), torch.tensor(adv_input_labels)))
        # self.current_sample_idx += 1
        self.read_nraws(self.labels[index])
        # print("In CharBertDataset: ", self.labels[index])
        # if len(self.examples) == 0 or self.current_sample_idx == len(self.examples):
        #     self.read_nraws(self.labels[index])
        #     self.current_sample_idx += 1

        return self.examples
    
class CombinedDataset(Dataset):
    def __init__(self, data, dataset1, dataset2):
        """
        Initialize the CombinedDataset with two datasets.
        :param dataset1: First dataset
        :param dataset2: Second dataset
        """
        self.data = data
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Ensure both datasets have the same length
        if len(dataset1) != len(dataset2):
            raise ValueError("Datasets must have the same length")
        self.file_paths = [k for k,v in data.items()]
        self.index_to_filename = {i: fname for i, fname in enumerate(self.file_paths)}

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.dataset1)

    def __getitem__(self, idx):
        """
        Return a combined item from both datasets.
        :param idx: Index of the item to fetch
        """
        # print("\n\n", idx)
        # idx = self.index_to_filename[idx]
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]

        item1.update(item2)

        return item1

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
        # print(folds)
        return folds
    
    def get_indices_iam():
        base_path = os.path.join(DATA_PROCESSED, "jh", "ground_truth", "jh_gt.json")
        with open(base_path, 'r') as file:
            data = json.load(file)
        file_list = list(data.keys())
        random.shuffle(file_list)

        total_files = len(file_list)
        train_end = int(total_files * 0.8)
        val_end = train_end + int(total_files * 0.1)

        train_files = file_list[:train_end]
        val_files = file_list[train_end:val_end]
        test_files = file_list[val_end:]

        return [[train_files, val_files, test_files]]

    # # Get indices for IAM
    # def get_indices_iam():
    #     fold = tuple()
    #     base_path = os.path.join(DATA_RAW, "IAM", "split")
    #     # Open the file in read mode
    #     for filename in ["trainset.txt", "validationset1.txt", "testset.txt"]:
    #         with open(os.path.join(base_path, filename), "r") as file:
    #             lines = file.readlines()
    #         lines = [i.rstrip("\n") for i in lines]
    #         fold = fold + (lines,)
    #     folds = [fold]
    #     return folds
    
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
    charbert_dataset = CharBertDataset(charbert_dataset_args, data=image, labels=gt)
    combined_dataset = CombinedDataset(gt, custom_dataset, charbert_dataset)
    # Create instances of my custom dataset for training, validation and testing purposes
    for index, fold in enumerate(folds):
        # Create DataLoader for training, validation, and test sets
        train_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(fold[0]))
        val_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(fold[1]))
        test_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(fold[2]))
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
    # bullinger_folds = get_split_indices("Bullinger", data["bullinger_image"], data["bullinger_gt"])
    # icfhr_folds = get_split_indices("ICFHR", data["icfhr_image"], data["icfhr_gt"])

    # Load the data loaders for all datasets
    gw_data_loaders = process_data_loader(data["GW_image"], data["GW_gt"], GW_folds, batch_size, processor, max_target_length)
    iam_data_loaders = process_data_loader(data["IAM_image"], data["IAM_gt"], IAM_folds, batch_size, processor, max_target_length)
    # bullinger_data_loaders = process_data_loader(data["bullinger_image"], data["bullinger_gt"], bullinger_folds, batch_size, processor, max_target_length, "Bullinger")
    # icfhr_data_loaders = process_data_loader(data["icfhr_image"], data["icfhr_gt"], icfhr_folds, batch_size, processor, max_target_length)
    
    bullinger_data_loaders, icfhr_data_loaders = [], []
    return gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders

if __name__=="__main__":
    data_loader_keys = cfg.trocr_charbert["experiment1"]["data_loader_keys"]
    gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(**data_loader_keys)
    test_loader = gw_data_loaders["cv1"]["train"]
    for batch in test_loader:
        for i in list(batch.keys()):
            print(i)
            if isinstance(batch[i], torch.Tensor):
                print(batch[i].size())
        # print(batch.keys())
        # Extract the image tensor from the batch (adjust based on your actual data structure)
        # images = batch[0]
        # Check if images is a tensor and has the expected shape (3 channels for RGB images)
        # if isinstance(images, torch.Tensor):
            # print("Images are present in the batch.")
        # else:
            # print("No images found in the batch.")
    # shutil.rmtree(os.path.join(".", "data", "raw", "Bullinger", "extracted_folder"))
    # train_en_loader, val_en_loader, test_en_loader = get_data_loader(512, 0.2)
    # # Iterate through the DataLoader
    # for batch_idx, (data, labels) in enumerate(train_en_loader):
    #     "data" contains input data, and "labels" contains corresponding labels
    #     print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {len(labels)}")