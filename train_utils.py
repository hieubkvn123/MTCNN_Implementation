import os
import cv2
import shutil
import tqdm
import numpy as np
import matplotlib.pyplot as plt

### Other dependencies ###
from PIL import Image
from dataloader.obj_detection import DataLoader

### Tensorflow dependencies ###
import tensorflow as tf
import tensorflow_addons as tfa
from custom_giou import GIoU
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy

bce  = BinaryCrossentropy(from_logits=True)
giou = GIoU(mode='giou', reg_factor=2e-4) # tfa.losses.GIoULoss()
opt = Adam(lr=0.00001, amsgrad=True)
accuracy = tf.keras.metrics.Accuracy()

def make_pnet_confidence_map(model, test_img_file, out_file, output_dir='pnet_conf_maps', threshold=0.6):
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir) # create dir if not exists 

    img = cv2.imread(test_img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - 127.5) / 127.5

    prediction = model.predict(np.array([img]))

    confidence = prediction[0][0]
    conf_map = confidence[:,:,1]
    conf_map = tf.sigmoid(conf_map).numpy()
    conf_map[conf_map >= threshold] = 255
    conf_map[conf_map < threshold] = 0
    conf_map = conf_map.astype(np.uint8)

    
    output_file = f'{output_dir}/{out_file}.png'
    plt.imshow(conf_map)
    plt.title(f'GIF #{out_file}')
    plt.savefig(output_file)

@tf.function
def train_step(model, batch, n_classes=10):
    with tf.GradientTape() as tape:
        img, (bbox, prob) = batch
        bbox = tf.expand_dims(bbox, axis=1)
        bbox = tf.expand_dims(bbox, axis=1)
        prob = tf.expand_dims(prob, axis=1)
        prob = tf.expand_dims(prob, axis=1)
        prob = tf.one_hot(prob, depth=n_classes)
        pr_prob, pr_bbox = model(img, training=True)
        acc = accuracy(tf.math.argmax(prob, axis=3), tf.math.argmax(pr_prob, axis=3))

        # print(pr_prob.shape, prob.shape)
        cls_loss = bce(prob, pr_prob)
        bbx_loss = giou(bbox, pr_bbox)

        loss = cls_loss + bbx_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    return cls_loss, bbx_loss, acc

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

