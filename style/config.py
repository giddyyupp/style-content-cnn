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
PYTORCH_MODELS = "../models/Style/m2_3/ills_crop150_padded/"

LOAD_TRAINED_MODEL = 0 # train asamasinda eski bir model yuklemek isterseniz bunu 1 yapin.

# train dataset
train_dir_path = '../dataset/ArtData/train'
train_path = train_dir_path + "/dataset_ill_v3/"

# test dataset
test_dir_path = '../dataset/ArtData/test'
test_path = test_dir_path + '/test_ill_v3_ganilla/'

# test_model name
MODEL_NAME = "../models/art_model-159.pkl"

