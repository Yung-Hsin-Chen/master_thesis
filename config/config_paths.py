import os

ROOT = "."

# config
CONFIG = os.path.join(ROOT, "config")
CONFIG_CONST = os.path.join(CONFIG, "config_const.ini")
PUNCTUATION_LIST = os.path.join(CONFIG, "punctuation_list.json")
CONFIG_LOGGING = os.path.join(CONFIG, "config.py")

# data
DATA = os.path.join(ROOT, "data")
DATA_PROCESSED = os.path.join(DATA, "processed")
DATA_RAW = os.path.join(DATA, "raw")

# results
RESULTS = os.path.join(ROOT, "results")
PIJ = os.path.join(RESULTS, "p_ij")

# models
MODELS = os.path.join(ROOT, "models")
