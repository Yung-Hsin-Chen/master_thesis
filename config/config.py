# config.py
import logging

def configure_logging(log_file_path=None):
    handlers = [logging.StreamHandler()]  # Always print logs to the console

    if log_file_path:
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=logging.INFO,
        format="[\"INFO\" - %(asctime)s]  %(message)s",
        datefmt="%d-%m-%Y %H:%M",
        handlers=handlers
    )
