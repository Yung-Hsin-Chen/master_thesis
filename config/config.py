# config.py
import logging

def configure_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="[\"INFO\" - %(asctime)s]  %(message)s",
        datefmt="%d-%m-%Y %H:%M",
        handlers=[
            logging.FileHandler(log_file_path),  # specify the log file
            logging.StreamHandler(),  # print logs to the console
        ]
    )
