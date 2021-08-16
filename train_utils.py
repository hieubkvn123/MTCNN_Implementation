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

bce  = BinaryCrossentropy(from_logits=False)
mse  = tf.keras.losses.MeanSquaredError()
giou = tfa.losses.GIoULoss(mode='giou') # GIoU(mode='giou', reg_factor=2e-4)
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
    conf_map = confidence[:,:,0]
    conf_map = 1 - tf.sigmoid(conf_map).numpy()
    conf_map[conf_map >= threshold] = 255
    conf_map[conf_map < threshold] = 0
    conf_map = conf_map.astype(np.uint8)

    
    output_file = f'{output_dir}/{out_file}.png'
    plt.imshow(conf_map)
    plt.title(f'GIF #{out_file}')
    plt.savefig(output_file)

@tf.function
def train_step(model, batch, box_reg='giou', n_classes=10):
    with tf.GradientTape() as tape:
        img, (bbox, prob) = batch
        bbox = tf.expand_dims(bbox, axis=1)
        bbox = tf.expand_dims(bbox, axis=1)
        prob = tf.expand_dims(prob, axis=1)
        prob = tf.expand_dims(prob, axis=1)
        prob = tf.one_hot(prob, depth=n_classes)
        pr_prob, pr_bbox = model(img, training=True)
        acc = accuracy(tf.math.argmax(prob, axis=3), tf.math.argmax(pr_prob, axis=3))

        cls_loss = bce(prob, pr_prob)

        if(box_reg == 'giou'):
            bbx_loss = giou(bbox, pr_bbox)
        else:
            bbx_loss = mse(bbox, pre_bbox)

        loss = cls_loss + bbx_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    return cls_loss, bbx_loss, acc

@tf.function
def validation_step(model, batch, n_classes=10):
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

def _overfitting(losses, patience):
    head = losses[-patience:]

    # Check if sorted head is equal to unsorted head
    return sorted(head) == head

def train(model, dataset, val_dataset, weights_file, logdir='logs', box_reg='giou', 
        n_classes=10, steps_per_epoch=1000, validation_steps=100, 
        epochs=100, patience=15, make_conf_map=False, early_stopping=False):
    '''
        Parameters:
        @model : The tensorflow-keras model to be trained
        @dataset : The training dataset (DataLoader)
        @val_dataset : The validation dataset (DataLoader)
        @weights_file : The destination file where model weights are stored
        @logdir : The tensorboard logging directory
        @box_reg : The bounding box regression method. 'giou' for GIoU and 'mse' for mean squared error
        @n_classes : The number of object classes
        @steps_per_epoch : Number of batches in the training dataset
        @validation_steps : Number of batches in the validation dataset
        @epochs : Number of training iterations
        @patience : Applicable for early stopping.
        @make_conf_map : Log confidence map in the classification task or not.
        @early_stopping : Whether to apply early stopping or not
    '''
    if(os.path.exists(weights_file)):
        print('Checkpoint exists, loading to model ... ')
        model.load_weights(weights_file)

    writer = SummaryWriter(log_dir=logdir)
    num_gif_files = 0
    summary = {'train_bbox_loss' : 0, 'train_cls_loss' : 0, 'train_acc' : 0,
            'val_bbox_loss': 0, 'val_cls_loss' : 0, 'val_acc' : 0}
    history = {
        'train' : {
            'cls' : [], 'bbox' : []
        },
        'val' : {
            'cls' : [], 'bbox' : []
        }
    }

    for i in range(epochs):
        print(f'Epoch {i+1}/{epochs}')
        with tqdm.tqdm(total=steps_per_epoch) as pbar:
            cls_losses = []
            box_losses = []
            accuracies = []
            for j in range(steps_per_epoch):
                batch = next(iter(dataset))

                cls_loss, bbox_loss, acc = train_step(model, batch, box_reg=box_reg, n_classes=n_classes)
                cls_loss = cls_loss.numpy()
                bbox_loss = bbox_loss.numpy()
                acc = acc.numpy()

                cls_losses.append(cls_loss)
                box_losses.append(bbox_loss)
                accuracies.append(acc)

                if((j+1) % 100 == 0 and make_conf_map):
                    num_gif_files += 1
                    make_pnet_confidence_map(model, 'test/test.png', num_gif_files)

                pbar.set_postfix({
                    'cls_loss': f'{np.array(cls_losses).mean():.4f}',
                    'bbox_loss' : f'{np.array(box_losses).mean():.4f}',
                    'accuracy' : f'{acc:.4f}'
                })
                pbar.update(1)

            summary['train_bbox_loss'] = np.array(box_losses).mean()
            summary['train_cls_loss'] = np.array(cls_losses).mean()
            summary['train_acc'] = np.array(accuracies).mean()

        print('Validating ... ')
        with tqdm.tqdm(total=validation_steps // 5, colour='green') as pbar:
            cls_losses = []
            box_losses = []
            accuracies = []

            for j in range(validation_steps // 5):
                batch = next(iter(val_dataset))
                cls_loss, bbox_loss, acc = validation_step(model, batch, n_classes=n_classes)
                cls_losses.append(cls_loss)
                box_losses.append(bbox_loss)
                accuracies.append(acc)

                pbar.set_postfix({
                    'cls_loss' : f'{np.array(cls_losses).mean():.4f}',
                    'bbox_loss' : f'{np.array(box_losses).mean():.4f}',
                    'accuracy' : f'{acc:.2f}'
                })
                pbar.update(1)

            summary['val_bbox_loss'] = np.array(box_losses).mean()
            summary['val_cls_loss'] = np.array(cls_losses).mean()
            summary['val_acc'] = np.array(accuracies).mean()
       
        # Log bounding box losses to tensorboard log dir
        writer.add_scalars('bounding_box_loss', {
            'train' : summary['train_bbox_loss'],
            'val' : summary['val_bbox_loss']
        }, i)

        # Log classification losses to tensorboard log dir
        writer.add_scalars('classification_loss', {
            'train' : summary['train_cls_loss'],
            'val' : summary['val_cls_loss']
        }, i)

        # Log accuracies to tensorboard log dir
        writer.add_scalars('accuracy', {
            'train' : summary['train_acc'],
            'val' : summary['val_acc']
        }, i)

        writer.flush()

        # Append information into the history log
        history['train']['cls'].append(summary['train_cls_loss'])
        history['train']['bbox'].append(summary['train_bbox_loss'])
        history['val']['cls'].append(summary['val_cls_loss'])
        history['val']['bbox'].append(summary['val_bbox_loss'])

        # Check if the model is currently overfitting
        if(len(history['val']['cls']) >= patience): # If patience if p, at least p epochs must be trained to consider early stopping
            if(_overfitting(history['val']['cls'], patience) or _overfitting(history['val']['bbox'], patience)):
                # If early stopping is set, break the training loop
                if(early_stopping):
                    break
    
        if(len(history['val']['cls']) >= 2):
            if(not _overfitting(history['val']['cls'], 2)):
                print('Saving model weights to ', weights_file, ' ... ')
                model.save_weights(weights_file)
        else:
            print('Saving model weights to ', weights_file, ' ... ')
            model.save_weights(weights_file)
