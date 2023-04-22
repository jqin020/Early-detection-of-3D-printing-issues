import os
import numpy as np
import argparse
import pytorch_lightning as pl
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *
import collections
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--seed", default=1234, type=int, help="Set seed for training"
)

args = parser.parse_args()
seed = args.seed

set_seed(seed)
os.environ['CHECKPOINT_PATH'] = '/home/npg4/Jingtao/kaggle/Early detection of 3D printing issues/3D-main/src/checkpoints/' \
                                '11042023/1234/MHResAttNet-dataset_full-11042023-epoch=07-val_loss=0.01-val_acc=1.00.ckpt'
model = ParametersClassifier.load_from_checkpoint(
    checkpoint_path=os.environ.get("CHECKPOINT_PATH"),
    num_classes=2,
    lr=INITIAL_LR,
    gpus=1,
    transfer=False,
)
model.eval()

data = ParametersDataModule(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    csv_file=DATA_CSV,
    test_csv_file=TEST_DATA_CSV,
    # image_dim=(320, 320),
    dataset_name=DATASET_NAME,
    mean=DATASET_MEAN,
    std=DATASET_STD,
    # transform=False,
)
data.setup('test')

# preds = []
# for batch_index,(x,y) in enumerate(data.test_dataloader()):
#     print(f'batch {batch_index}')
#     _,batch_pred = torch.max(model(x),1)
#     preds.append(batch_pred.cpu())

# model_preds = list(torch.cat(preds).numpy())
# torch.save(model_preds,'model_preds.pt')
model_preds = torch.load('model_preds.pt')

# without print id
dataframe_submission = data.test_dataset.dataframe
dataframe_submission['has_under_extrusion'] = model_preds
dataframe_submission = dataframe_submission.drop(['printer_id','print_id'],axis=1)
dataframe_submission.to_csv(f'submission.csv',index=False)


# with print id
dataframe_submission = data.test_dataset.dataframe
img_paths = list(dataframe_submission['img_path'])
printer_ids = list(dataframe_submission['printer_id'])
print_ids = list(dataframe_submission['print_id'])
num_files = len(img_paths)

counter = collections.defaultdict(int)
total_counter = collections.defaultdict(int)
for idx in range(num_files):
    exp_id = f'{printer_ids[idx]}-{print_ids[idx]}'
    total_counter[exp_id] += 1
    if model_preds[idx] == 1:
        counter[exp_id] += 1
    # else:
        # counter[exp_id] -= 1

filtered_preds = []
print_extrusion_rate = collections.defaultdict(int)
for idx in range(num_files):
    exp_id = f'{printer_ids[idx]}-{print_ids[idx]}'
    print_extrusion_rate[exp_id] = counter[exp_id]/total_counter[exp_id]

    if print_extrusion_rate[exp_id] > 0.1:
        filtered_preds.append(1)
    elif print_extrusion_rate[exp_id] < 0.1:
        filtered_preds.append(0)


dataframe_submission['has_under_extrusion'] = filtered_preds
dataframe_submission = dataframe_submission.drop(['printer_id','print_id'],axis=1)
dataframe_submission.to_csv(f'submission_withprintid.csv',index=False)
