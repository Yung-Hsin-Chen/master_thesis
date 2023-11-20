import os
import json

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