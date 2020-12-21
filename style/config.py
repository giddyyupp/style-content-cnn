import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '../dataset/')


CLASS_LIST = ["axel", "korky", "polacco", "mbrown", "dmckee", "khenkes", "rcurto", "tross", "serap",  "scart", "photo"]

CLASS_COUNT = len(CLASS_LIST)
BATCH_SIZE = 32
EPOCH_COUNT = 400
SAVE_PERIOD_IN_EPOCHS = 5
LOG_STEP = 100
LEARNING_RATE = 2.5e-4
NUM_WORKERS = 32

USE_PATCH = 1
PATCH_SIZE = 150
RESIZE = 224

# model save dir
PYTORCH_MODELS = "../models/Style/"

LOAD_TRAINED_MODEL = 0 # if you want to load a model during train change this to 1.

# train dataset
train_dir_path = '../dataset_style/train'
train_path = train_dir_path + "/"

# test dataset
test_dir_path = '../dataset_style/test'
test_path = test_dir_path + '/'

# test_model name
MODEL_NAME = "../models/Style/art_model-159.pkl"

