import os

RANDOM_SEED = 51

# get the current working directory
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# path to data relative to the parent directory
DATA_PATH = os.path.join(PARENT_DIR, "data")
