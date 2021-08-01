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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy

### Some constants ###
# weights_dir = 'road_signs_1'
weights_dir = 'road_signs_w_dataloader_new'
onet_tensorboard_logdir = 'onet_logs'
onet_weights = f'weights/{weights_dir}/onet.weights.hdf5'
onet_configs = f'weights/{weights_dir}/onet.json'
train_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/train"
test_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/test"

input_dim = 12 # 48
epochs = 100 # 500
batch_size = 16

if(not os.path.exists(f'weights/{weights_dir}')):
    print('[INFO] Created weight directory ...')
    os.mkdir(f'weights/{weights_dir}')
    
if(os.path.exists(onet_tensorboard_logdir)):
    print('[INFO] Clearing O-Net log directory ... ')
    shutil.rmtree(onet_tensorboard_logdir)

### Loading dataset ###
### Creating the train loader ###
train_loader = DataLoader(train_dir, format_='darknet',
                    color_space='rgb', img_size=input_dim*4, batch_size=64,
                   crop_to_bounding_box=False)

### Creating the test loader ###
test_loader = DataLoader(test_dir, format_='darknet',
                    color_space='rgb', img_size=input_dim*4, batch_size=64,
                   crop_to_bounding_box=False)
train_dataset = train_loader.get_train_dataset()
val_dataset = train_loader.get_val_dataset()

n_classes = train_loader.n_classes
configs = {
    'input_shape' : input_dim*4,
    'batch_norm' : True,
    'dropout' : True,
    'n_classes' : n_classes
}
onet = build_pnet_model(input_shape=configs['input_shape'], batch_norm=configs['batch_norm'], dropout=configs['dropout'],
                        n_classes=configs['n_classes'], l2_norm=True)
print(f'[INFO] Storing O-Net configuration to {onet_configs}')
with open(onet_configs, 'w') as config_file:
    json.dump(configs, config_file, indent=4, sort_keys=True)

print(onet.summary())

### Define training loop and start training ###
steps_per_epoch = train_loader.dataset_len
validation_steps = train_loader.val_len
bce  = CategoricalCrossentropy(from_logits=False) # BinaryCrossentropy(from_logits=False)
giou = GIoU(mode='giou', reg_factor=2e-4) # tfa.losses.GIoULoss()
opt = Adam(lr=0.00001, amsgrad=True)
accuracy = tf.keras.metrics.Accuracy()

train(onet, train_dataset, val_dataset, onet_weights, 
        logdir=onet_tensorboard_logdir,
        n_classes=n_classes, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps, 
        epochs=epochs, 
        make_conf_map=False)
print('[INFO] Training halted, plotting training history ... ')
