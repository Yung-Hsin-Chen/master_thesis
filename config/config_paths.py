import os

ROOT = "."

# config
CONFIG = os.path.join(ROOT, "config")
CONFIG_CONST = os.path.join(CONFIG, "config_const.ini")
CONFIG_JSON = os.path.join(CONFIG, "config.json")
CONFIG_LOGGING = os.path.join(CONFIG, "config.py")

# data
DATA = os.path.join(ROOT, "data")
DATA_PROCESSED = os.path.join(DATA, "processed")
DATA_RAW = os.path.join(DATA, "raw")

# src
SRC = os.path.join(ROOT, "src")
PROCESSOR = os.path.join(SRC, "processor")
TRAIN = os.path.join(SRC, "train")
