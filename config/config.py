import logging
import os

def get_my_logger(name, log_file_path=None):
    # Create a custom logger
    logger = logging.getLogger(name)
    # Clear previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.propagate = False  # Prevent the log messages from being propagated to the root logger

    # Set the level of your logger
    logger.setLevel(logging.INFO)  # or DEBUG, depending on your need

    # Create formatters
    log_format = '[\"%(levelname)s\" - %(asctime)s]  %(message)s'
    # log_format = '[\"INFO\" - %(asctime)s]  %(message)s'
    date_format = "%d-%m-%Y %H:%M"

    # Create and add the console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # Create and add the file handler, if a file path is provided
    if log_file_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    return logger

def configure_logging(script_name="MyScript", log_file_path=None):
    # Get a custom logger for your script
    logger = get_my_logger(script_name, log_file_path)
    return logger
