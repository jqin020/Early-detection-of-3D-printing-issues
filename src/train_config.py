import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms


DATE = datetime.now().strftime("%d%m%Y")

dataset_switch = 1

os.environ['DATA_DIR'] = '/home/npg4/Jingtao/kaggle/Early detection of 3D printing issues/3D-main'
DATA_DIR = os.environ.get("DATA_DIR",'not found')

#
if dataset_switch == 0:
    DATASET_NAME = "dataset_single_layer"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/caxton_dataset_filtered_single.csv",
    )
    DATASET_MEAN = [0.16853632, 0.17632364, 0.10495131]
    DATASET_STD = [0.05298341, 0.05527821, 0.04611006]
elif dataset_switch == 1:
    DATASET_NAME = "dataset_full"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "dataset/train.csv",
    )
    TEST_DATA_CSV = os.path.join(
        DATA_DIR,
        'dataset/test.csv',
    )

    DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
    DATASET_STD = [0.066747, 0.06885352, 0.07679665]

elif dataset_switch == 2:
    DATASET_NAME = "dataset_equal"
    DATA_CSV = os.path.join(
        DATA_DIR,
        "caxton_dataset/caxton_dataset_filtered_equal.csv",
    )

    DATASET_MEAN = [0.2925814, 0.2713622, 0.14409496]
    DATASET_STD = [0.0680447, 0.06964592, 0.0779964]

INITIAL_LR = 0.001

BATCH_SIZE = 512 #32 for training 512 for testing
MAX_EPOCHS = 10

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "ddp"


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.2915257, 0.27048784, 0.14393276],
            [0.2915257, 0.27048784, 0.14393276],
        )
    ],
)