DATA_PATH = "dataset"
TRAIN_PATH ='/train'
VAL_PATH ='/val'
TEST_PATH='/test'

NUM_OF_FEATURES = 5
RANDOM_SEED = 42
NUM_OF_EPOCHS = 100
BATCH_SIZE = 16
MAX_LEARNING_RATE = 0.01
GRAD_CLIP = 0.1
WEIGHT_DECAY = 1e-4
IMAGE_CHANEL = 3
CHECKPOINT_PATH = 'model/BoneClassification.pth'
IMAGE_SIZE = 512

CLS = [
    'abdominal',
    'adult',
    'others',
    'pediatric',
    'spine'
]