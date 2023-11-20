from src.data.data_loader import *
import pytest
import json
import os

def test_get_data_loader():
    with open(os.path.join(".", "config", "config_const.yaml"), "r") as config_file:
        resize_data = yaml.safe_load(config_file)["training"]
    batch_size, test_size = resize_data["batch_size"], resize_data["test_size"]
    resize_height, resize_width = resize_data["resize_height"], resize_data["resize_width"]
    # Assert data types and value range 
    assert isinstance(batch_size, int)
    assert isinstance(test_size, float)
    assert isinstance(resize_height, int)
    assert isinstance(resize_width, int)
    assert 0 < test_size < 1
    # Get data loaders
    train_en_loader, val_en_loader, test_en_loader = get_data_loader(batch_size, test_size)
    for batch_idx, (data, labels) in enumerate(train_en_loader):
        if batch_idx!=len(train_en_loader)-1:
            assert data.size() == torch.Size([batch_size, 1, resize_height, resize_width])
            assert len(labels) == batch_size
    return