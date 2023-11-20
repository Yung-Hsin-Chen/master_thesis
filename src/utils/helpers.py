import h5py
import os

# Store processed data dictionary into HDF5
def store_processed_data(data_name: str, data: dict) -> None:
    """
    Store the processed data dictionary into HDF5 in data/processed.

    Parameters
    -----------
    data_name: str
        # TODO add German dataset here
        The name of the data, can be "GW", "IAM" or "" with "_gt" or "_image" as the ending.
    data: dict
        The keys are the indices, while the values are the corresponding ground truth text or image path.

    Returns
    --------
    None
    """
    # TODO include the GERMAN dataset name in docstring
    processed_path = os.path.join(".", "data", "processed", data_name[:data_name.find("_")], "ground_truth")
    # Ensure the folder exists; create it if it doesn't
    os.makedirs(processed_path, exist_ok=True)
    # Store the processed data dictionary as h5
    file_name = os.path.join(processed_path, data_name+".h5")
    with h5py.File(file_name, "w") as h5_file:
        for key, value in data.items():
            h5_file[key] = value
    return