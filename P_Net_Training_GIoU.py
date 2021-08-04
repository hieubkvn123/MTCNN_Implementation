import os
import cv2
import json
import tqdm
import time
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

### Other dependencies ###
from PIL import Image
from dataloader.obj_detection import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

### Tensorflow dependencies ###
import tensorflow as tf
import tensorflow_addons as tfa
from custom_giou import GIoU
from train_utils import train
from models import build_pnet_model
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy

### Some constants ###
# weights_dir = 'road_signs_w_dataloadr_l2norm'
weights_dir = 'chest_xray'
pnet_tensorboard_logdir = 'pnet_logs'
pnet_weights = f'weights/{weights_dir}/pnet.weights.hdf5'
pnet_configs = f'weights/{weights_dir}/pnet.json'
#train_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/obj/train"
#val_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/obj/val"

train_dir = "/home/minhhieu/Desktop/Hieu/datasets/ChestXRay_Cropped/images/train"
val_dir = "/home/minhhieu/Desktop/Hieu/datasets/ChestXRay_Cropped/images/val"

input_dim = 12*4 # 48
epochs = 100 # 500
batch_size = 16

if(not os.path.exists(f'weights/{weights_dir}')):
    print('[INFO] Created weight directory ...')
    os.mkdir(f'weights/{weights_dir}')
    
if(os.path.exists(pnet_tensorboard_logdir)):
    print('[INFO] Clearing P-Net log directory ... ')
    shutil.rmtree(pnet_tensorboard_logdir)
    
### Loading dataset ###
### Creating the train loader ###
train_loader = DataLoader(train_dir, format_='darknet', preprocess='standard',
                    color_space='rgb', img_size=input_dim, batch_size=16,
                   crop_to_bounding_box=False)

### Creating the test loader ###
val_loader = DataLoader(val_dir, format_='darknet', preprocess='standard',
                    color_space='rgb', img_size=input_dim, batch_size=16,
                   crop_to_bounding_box=False)

train_dataset = train_loader.get_train_dataset()
val_dataset = val_loader.get_train_dataset()
steps_per_epoch = train_loader.dataset_len
validation_steps = val_loader.dataset_len

n_classes = train_loader.n_classes
print(n_classes)
configs = {
    'input_shape' : input_dim,
    'batch_norm' : True,
    'dropout' : True,
    'n_classes' : n_classes
}

print(f'[INFO] Storing P-Net configuration to {pnet_configs}')
with open(pnet_configs, 'w') as config_file:
    json.dump(configs, config_file, indent=4, sort_keys=True)

### Experimenting with l2 normalization on the final logit layer ###
pnet = build_pnet_model(input_shape=configs['input_shape'], batch_norm=configs['batch_norm'], dropout=configs['dropout'],
                        n_classes=configs['n_classes'], l2_norm=True)
print(pnet.summary())

### Start training ###
train(pnet, train_dataset, val_dataset, pnet_weights, 
        logdir=pnet_tensorboard_logdir,
        n_classes=n_classes, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps, 
        epochs=epochs, 
        make_conf_map=True)
print('[INFO] Training halted, plotting training history ... ')
