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
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy

### Implement the P-Net architecture ###
def conv_block(in_filters, out_filters, kernel_size=3, batch_norm=False):
    inputs = Input(shape=(None, None, in_filters))
    p_layer = Conv2D(out_filters, kernel_size=kernel_size, strides=(1, 1), padding="valid", kernel_regularizer=l1(2e-4))(inputs)
    if(batch_norm) : p_layer = BatchNormalization()(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)

    block = Model(inputs = inputs, outputs=p_layer)
    return block

def res_block(in_filters, out_filters, kernel_size=3, downsample=False):
    inputs = Input(shape=(None, None, in_filters))
    identity = inputs

    conv1 = Conv2D(out_filters, kernel_size=kernel_size, strides=(1,1), padding='same', kernel_regularizer=l1(2e-4))(inputs)
    bn1   = BatchNormalization()(conv1)
    relu1 = PReLU(shared_axes=[1, 2])(bn1)

    conv2 = Conv2D(out_filters, kernel_size=kernel_size, strides=(1,1), padding='same', kernel_regularizer=l1(2e-4))(inputs)
    bn2   = BatchNormalization()(conv2)
    relu2 = PReLU(shared_axes=[1,2])(bn2)

    if(downsample):
        identity = Conv2D(out_filters, kernel_size=1, strides=(1,1), padding='same', kernel_regularizer=l1(1e-4))(identity)
        identity = BatchNormalization()(identity)

    res_out = relu2 + identity
    res_out = ReLU()(res_out)

    block = Model(inputs=inputs, outputs=res_out)
    return block

def res_p_layer_block(in_filters, out_filters, n_res_blk=1, kernel_size=3, downsample=False, batch_norm=False):
    inputs = Input(shape=(None, None, in_filters))
    res_out = res_block(in_filters, out_filters, kernel_size=kernel_size, downsample=downsample)(inputs)
    
    if(n_res_blk > 1):
        for i in range(n_res_blk-1):
            res_out = res_block(out_filters, out_filters, kernel_size=kernel_size, downsample=downsample)(res_out)

    p_layer = Conv2D(out_filters, kernel_size=kernel_size, strides=(1, 1), padding="valid", kernel_regularizer=l1(2e-4))(res_out)
    if(batch_norm) : p_layer = BatchNormalization()(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)
    block = Model(inputs=inputs, outputs=p_layer)

    return block

def build_pnet_model(input_shape=None, batch_norm=True, dropout=False, l2_norm=False, n_classes=2, activation='relu'):
    if(input_shape is not None):
        if(input_shape not in [12, 24, 48, 112, 224]):
            raise Exception('Input shape must be in 12, 24, 48, 112 or 224')

    inputs = Input(shape=(None, None, 3))
    p_layer = conv_block(3, 10, kernel_size=3, batch_norm=batch_norm)(inputs)

    if(input_shape is not None):
        if(input_shape >= 24):
            p_layer = conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

        if(input_shape >= 48):
            p_layer = conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

        if(input_shape >= 112):
            p_layer = conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l1(2e-4))(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l1(2e-4))(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)
    if(dropout) : p_layer = Dropout(0.5)(p_layer)

    if(l2_norm):
        p_layer = Lambda(lambda x : K.l2_normalize(x, axis=3))(p_layer)

    p_layer_out1 = Conv2D(n_classes, kernel_size=(1, 1), strides=(2, 2), kernel_regularizer=l1(2e-4), name='prob_logits')(p_layer)
    p_layer_out1 = Softmax(axis=3, name='probability')(p_layer_out1)
    p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(2, 2), activation='sigmoid', name='bbox_regression')(p_layer)

    p_net = Model(inputs, [p_layer_out1, p_layer_out2], name='P-Net')

    return p_net

# A variant of P-Net that has residual block from ResNet
def build_residual_pnet_model(input_shape=None, batch_norm=True, dropout=False, l2_norm=False, n_classes=2, activation='relu'):
    if(input_shape is not None):
        if(input_shape not in [12, 24, 48, 112, 224]):
            raise Exception('Input shape must be in 12, 24, 48, 112 or 224')

    inputs = Input(shape=(None, None, 3))
    p_layer = res_p_layer_block(3, 10, kernel_size=3, batch_norm=batch_norm, downsample=True)(inputs)

    if(input_shape is not None):
        if(input_shape >= 24):
            p_layer = res_p_layer_block(10, 10, kernel_size=3, batch_norm=batch_norm, downsample=True)(p_layer)

        if(input_shape >= 48):
            p_layer = res_p_layer_block(10, 10, kernel_size=3, batch_norm=batch_norm, downsample=True)(p_layer)

        if(input_shape >= 112):
            p_layer = res_p_layer_block(10, 10, kernel_size=3, batch_norm=batch_norm, downsample=True)(p_layer)

    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l1(2e-4))(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l1(2e-4))(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)
    if(dropout) : p_layer = Dropout(0.5)(p_layer)

    if(l2_norm):
        p_layer = Lambda(lambda x : K.l2_normalize(x, axis=3))(p_layer)

    p_layer_out1 = Conv2D(n_classes, kernel_size=(1, 1), strides=(2, 2), kernel_regularizer=l1(2e-4), name='prob_logits')(p_layer)
    p_layer_out1 = Softmax(axis=3, name='probability')(p_layer_out1)
    p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(2, 2), activation='sigmoid', name='bbox_regression')(p_layer)

    p_net = Model(inputs, [p_layer_out1, p_layer_out2], name='P-Net')

    return p_net

