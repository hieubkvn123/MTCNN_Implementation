import os
import cv2
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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy

### Some constants ###
input_dim = 12 # 48
# weights_dir = 'road_signs_1'
weights_dir = 'road_signs_w_dataloader'
pnet_tensorboard_logdir = 'pnet_logs'
rnet_tensorboard_logdir = 'rnet_logs'
onet_tensorboard_logdir = 'onet_logs'

pnet_weights = f'weights/{weights_dir}/pnet.weights.hdf5'
rnet_weights = f'weights/{weights_dir}/rnet.weights.hdf5'
onet_weights = f'weights/{weights_dir}/onet.weights.hdf5'

rnet_configs = f'weights/{weights_dir}/rnet.json'

if(not os.path.exists(f'weights/{weights_dir}')):
    print('[INFO] Created weight directory ...')
    os.mkdir(f'weights/{weights_dir}')
    
if(os.path.exists(pnet_tensorboard_logdir)):
    print('[INFO] Clearing P-Net log directory ... ')
    shutil.rmtree(pnet_tensorboard_logdir)

if(os.path.exists(rnet_tensorboard_logdir)):
    print('[INFO] Clearing R-Net log directory ... ')
    shutil.rmtree(rnet_tensorboard_logdir)

if(os.path.exists(onet_tensorboard_logdir)):
    print('[INFO] Clearing O-Net log directory ... ')
    shutil.rmtree(onet_tensorboard_logdir)
    
epochs = 100 # 500
batch_size = 16
pnet_tensorboard = TensorBoard(log_dir=pnet_tensorboard_logdir)
pnet_checkpoint = ModelCheckpoint(pnet_weights, save_weights_only=True)
pnet_callbacks = [pnet_tensorboard, pnet_checkpoint]

rnet_tensorboard = TensorBoard(log_dir=rnet_tensorboard_logdir)
rnet_checkpoint = ModelCheckpoint(rnet_weights, save_weights_only=True)
rnet_callbacks = [rnet_tensorboard, rnet_checkpoint]

onet_tensorboard = TensorBoard(log_dir=onet_tensorboard_logdir)
onet_checkpoint = ModelCheckpoint(onet_weights, save_weights_only=True)
onet_early_stop1 = EarlyStopping(monitor='val_probability_loss', patience=15, verbose=1)
onet_early_stop2 = EarlyStopping(monitor='val_bbox_regression_loss', patience=15, verbose=1)
onet_callbacks = [onet_tensorboard, onet_checkpoint]

train_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/train"
test_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSRB/outputs/test"

### Loading dataset ###
### Creating the train loader ###
train_loader = DataLoader(train_dir, format_='darknet',
                    color_space='rgb', img_size=input_dim*2, batch_size=64,
                   crop_to_bounding_box=False)

### Creating the test loader ###
test_loader = DataLoader(test_dir, format_='darknet',
                    color_space='rgb', img_size=input_dim*2, batch_size=64,
                   crop_to_bounding_box=False)
train_dataset = train_loader.get_train_dataset()
val_dataset = train_loader.get_val_dataset()

### Implement the P-Net architecture ###
def conv_block(in_filters, out_filters, kernel_size=3, batch_norm=False):
    inputs = Input(shape=(None, None, in_filters))
    p_layer = Conv2D(out_filters, kernel_size=kernel_size, strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(inputs)
    if(batch_norm) : p_layer = BatchNormalization()(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)

    block = Model(inputs = inputs, outputs=p_layer)
    return block

def build_pnet_model(input_shape=None, batch_norm=True, dropout=False, n_classes=2, activation='relu'):
    if(input_shape is not None):
        if(input_shape not in [12, 24, 48, 112, 224]):
            raise Exception('Input shape must be in 12, 24, 48')

    inputs = Input(shape=(None, None, 3))
    p_layer = conv_block(3, 10, kernel_size=3, batch_norm=batch_norm)(inputs)

    if(input_shape is not None):
        if(input_shape >= 24):
            p_layer = conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

    if(input_shape is not None):
        if(input_shape >= 48):
            p_layer = conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

    if(input_shape is not None):
        if(input_shape >= 112):
            p_layer = conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)

    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(p_layer)
    p_layer = PReLU(shared_axes=[1, 2])(p_layer)
    if(dropout) : p_layer = Dropout(0.5)(p_layer)

    p_layer_out1 = Conv2D(n_classes, kernel_size=(1, 1), strides=(2, 2))(p_layer)
    p_layer_out1 = Softmax(axis=3, name='probability')(p_layer_out1)
    p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(2, 2), activation='sigmoid', name='bbox_regression')(p_layer)

    p_net = Model(inputs, [p_layer_out1, p_layer_out2], name='P-Net')

    return p_net


### GIoU formula ###
def GIoU(bboxes_1, bboxes_2, regularization=False):
    # 1. calulate intersection over union
    area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
    area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])

    intersection_wh = tf.minimum(bboxes_1[:, :, 2:], bboxes_2[:, :, 2:]) - tf.maximum(bboxes_1[:, :, :2], bboxes_2[:, :, :2])
    intersection_wh = tf.maximum(intersection_wh, 0)

    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = (area_1 + area_2) - intersection

    ious = intersection / tf.maximum(union, 1e-10)

    # 2. (C - (A U B))/C
    C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])
    C_wh = tf.maximum(C_wh, 0.0)
    C = C_wh[..., 0] * C_wh[..., 1]

    # 3. Additional regularization - to preserve aspect ratio
    lambda_ = 2e-4
    w_reg = lambda_ * K.binary_crossentropy(bboxes_1[..., 2], bboxes_2[...,2])
    h_reg = lambda_ * K.binary_crossentropy(bboxes_1[..., 3], bboxes_2[...,3])

    giou = ious - (C - union) / tf.maximum(C, 1e-10)
    if(regularization):
        giou += 0.5 * (w_reg + h_reg)

    return giou

n_classes = train_loader.n_classes
configs = {
    'input_shape' : input_dim*2,
    'batch_norm' : True,
    'dropout' : True,
    'n_classes' : n_classes
}
rnet = build_pnet_model(input_shape=configs['input_shape'], batch_norm=configs['batch_norm'], dropout=configs['dropout'],
                        n_classes=configs['n_classes'])
print(f'[INFO] Storing R-Net configuration to {pnet_configs}')
with open(rnet_configs, 'w') as config_file:
    json.dump(configs, config_file, indent=4, sort_keys=True)

print(rnet.summary())

### Define training loop and start training ###
steps_per_epoch = train_loader.dataset_len
validation_steps = train_loader.val_len
bce  = CategoricalCrossentropy(from_logits=False) # BinaryCrossentropy(from_logits=False)
giou = tfa.losses.GIoULoss()
opt = Adam(lr=0.0001, amsgrad=True)
accuracy = tf.keras.metrics.Accuracy()

@tf.function
def train_step(model, batch):
    with tf.GradientTape() as tape:
        img, (bbox, prob) = batch
        bbox = tf.expand_dims(bbox, axis=1)
        bbox = tf.expand_dims(bbox, axis=1)
        prob = tf.expand_dims(prob, axis=1)
        prob = tf.expand_dims(prob, axis=1)
        prob = tf.one_hot(prob, depth=n_classes)
        pr_prob, pr_bbox = model(img, training=True)
        
        # print(pr_prob.shape, prob.shape)
        cls_loss = bce(prob, pr_prob)
        bbx_loss = giou(bbox, pr_bbox)

        loss = cls_loss + bbx_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    return cls_loss, bbx_loss

@tf.function
def validation_step(model, batch):
    img, (bbox, prob) = batch
    bbox = tf.expand_dims(bbox, axis=1)
    bbox = tf.expand_dims(bbox, axis=1)
    prob = tf.expand_dims(prob, axis=1)
    prob = tf.expand_dims(prob, axis=1)
    prob = tf.one_hot(prob, depth=n_classes)
    pr_prob, pr_bbox = model(img, training=False)

    bbx_loss = giou(bbox, pr_bbox)
    cls_loss = bce(prob, pr_prob)
    acc = accuracy(tf.math.argmax(prob, axis=3), tf.math.argmax(pr_prob, axis=3))

    return cls_loss, bbx_loss, acc

def train(model, dataset, val_dataset, weights_file, steps_per_epoch=1000, validation_steps=100, epochs=100):
    if(os.path.exists(weights_file)):
        print('Checkpoint exists, loading to model ... ')
        model.load_weights(weights_file)

    for i in range(epochs):
        print(f'Epoch {i+1}/{epochs}')
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            for j in range(steps_per_epoch):
                batch = next(iter(dataset))

                cls_loss, bbox_loss = train_step(model, batch)
                cls_loss = cls_loss.numpy()
                bbox_loss = bbox_loss.numpy()

                # if((j + 1) % 100 == 0):
                #     print(f'[*] Batch #{j+1}, Epoch #{i+1}: Classification loss = {cls_loss:.4f}, BBox loss = {bbox_loss:.4f}')

                pbar.set_postfix({
                    'batch_id' : j+1,
                    'cls_loss': f'{cls_loss:.4f}',
                    'bbox_loss' : f'{bbox_loss:.4f}'
                })
                pbar.update(1)

        print('Saving model weights ... ')
        model.save_weights(weights_file)
        print('Validating ... ')
        with tqdm.tqdm(total=validation_steps // 5, colour='green') as pbar:
            for j in range(validation_steps // 5):
                batch = next(iter(val_dataset))
                cls_loss, bbox_loss, acc = validation_step(model, batch)

                pbar.set_postfix({
                    'batch_id' : j + 1,
                    'cls_loss' : f'{cls_loss:.4f}',
                    'bbox_loss' : f'{bbox_loss:.4f}',
                    'accuracy' : f'{acc:.2f}'
                })
                pbar.update(1)

train(rnet, train_dataset, val_dataset, rnet_weights, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=40)
print('[INFO] Training halted, plotting training history ... ')
