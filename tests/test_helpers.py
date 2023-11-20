from src.utils.helpers import *
import pytest
from unittest.mock import mock_open, patch
import json
import os

def test_store_processed_data(tmp_path):
    # Load the test data
    test_data_path = os.path.join(".", "tests", "mock_data.json")
    with open(test_data_path, "r") as json_file:
        test_data = json.load(json_file)["data"]

    # Call the function with the temporary folder and file path
    file_name = "test_output"
    folder_name = "test_folder"
    file_dir = os.path.join(tmp_path, folder_name)
    store_processed_data(file_name, test_data, file_dir)

    # Assert that the file was created in the correct folder
    assert file_name+".json" in os.listdir(file_dir)

    # Assert the file content
    # Open the HDF5 file and assert its content
    file_path = os.path.join(file_dir, file_name+".json")
    with open(file_path, "r") as json_file:
        loaded_dict = json.load(json_file)
        # Assert that the stored data matches the expected data
        assert loaded_dict == test_data
