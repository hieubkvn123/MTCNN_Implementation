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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy

### Specify the training directory and validating directory ###
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model-name', type=str, required=True, help='Model name is the directory to store the weights file')
parser.add_argument('--train-dir', type=str, required=True, help='Path the the Darknet annotated training dataset')
parser.add_argument('--val-dir', type=str, required=True, help='Path to the Darknet annotated validation dataset')
args = vars(parser.parse_args())

### Some constants ###
weights_dir = args['model_name'] 
train_dir = args['train_dir']
val_dir = args['val_dir']
rnet_tensorboard_logdir = 'rnet_logs'
rnet_weights = f'weights/{weights_dir}/rnet.weights.hdf5'
rnet_configs = f'weights/{weights_dir}/rnet.json'

#train_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/obj/train"
#val_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/obj/val"
#train_dir = "/home/minhhieu/Desktop/Hieu/datasets/DOTA/train_yolo/train"
#val_dir = "/home/minhhieu/Desktop/Hieu/datasets/DOTA/train_yolo/val"

input_dim = 12 # 48
epochs = 100 # 500
batch_size = 16

if(not os.path.exists(f'weights/{weights_dir}')):
    print('[INFO] Created weight directory ...')
    os.mkdir(f'weights/{weights_dir}')
    
    if(os.path.exists(rnet_tensorboard_logdir)):
        print('[INFO] Clearing R-Net log directory ... ')
        shutil.rmtree(rnet_tensorboard_logdir)

### Loading dataset ###
### Creating the train loader ###
train_loader = DataLoader(train_dir, format_='darknet', annot_format='corners',
                    color_space='rgb', img_size=input_dim*2, batch_size=16,
                   crop_to_bounding_box=False)

### Creating the val loader ###
val_loader = DataLoader(val_dir, format_='darknet', annot_format='corners',
                    color_space='rgb', img_size=input_dim*2, batch_size=16,
                   crop_to_bounding_box=False)
train_dataset = train_loader.get_train_dataset()
val_dataset = val_loader.get_train_dataset()

n_classes = train_loader.n_classes
configs = {
    'input_shape' : input_dim*2,
    'batch_norm' : True,
    'dropout' : True,
    'n_classes' : n_classes,
    'initial_epoch' : 0
}
rnet = build_pnet_model(input_shape=configs['input_shape'], batch_norm=configs['batch_norm'], dropout=configs['dropout'],
                        n_classes=configs['n_classes'], l2_norm=True)

if(not os.path.exists(rnet_configs)):
    print(f'[INFO] Storing R-Net configuration to {rnet_configs}')
    with open(rnet_configs, 'w') as config_file:
        json.dump(configs, config_file, indent=4, sort_keys=True)
else:
    # Reload the configs
    f = open(rnet_configs, 'r')
    configs = json.load(f)
    f.close()

print(rnet.summary())

### Define training loop and start training ###
steps_per_epoch = train_loader.dataset_len
validation_steps = val_loader.dataset_len

while(True):
    try:
        last_epoch = train(rnet, train_dataset, val_dataset, rnet_weights, 
                logdir=rnet_tensorboard_logdir,
                n_classes=n_classes, 
                box_reg='mse',
                steps_per_epoch=steps_per_epoch, 
                validation_steps=validation_steps, 
                epochs=epochs, 
                make_conf_map=False)
        
        # Save last epoch as initial epoch to configs
        configs['initial_epoch'] = last_epoch
        with open(rnet_configs, 'w') as config_file:
            print('[INFO] Saving configs ...')
            json.dump(configs, config_file, indent=4, sort_keys=True)
    except KeyboardInterrupt:
        break
