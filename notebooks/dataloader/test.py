import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import obj_detection
from sklearn.preprocessing import LabelEncoder
from dataset import DataLoader

directory = '/home/hieu/Downloads/dogs-vs-cats/train/train'
img_paths = glob.glob(os.path.join(directory, '*.jpg'))
labels = [x.split('/')[-1].split('.')[0] for x in img_paths]
labels = LabelEncoder().fit_transform(labels)

### Testing in dataset with label case ###
loader = DataLoader(directory, labels=labels, batch_size=64, random_noise=True, color_space='hsv', one_hot=True)
loader.parse_to_tfrecord('data_with_label.tfrecord')
dataset = loader.get_train_dataset()
x, y = next(iter(dataset))
x = ((x.numpy() * 127.5) + 127.5).astype(np.uint8)
y = y.numpy()

fig, ax = plt.subplots(8,8, figsize=(20,20))
for i in range(64):
    row = i // 8
    col = i %  8

    ax[row][col].imshow(x[i])
    ax[row][col].set_title(y[i])

plt.show()

### Testing in dataset without label case ###
loader = DataLoader(directory, labels=None, batch_size=64, random_noise=True)
loader.parse_to_tfrecord('data_without_label.tfrecord')
dataset = loader.get_train_dataset()
x = next(iter(dataset))
x = ((x.numpy() * 127.5) + 127.5).astype(np.uint8)

fig, ax = plt.subplots(8,8, figsize=(20,20))
for i in range(64):
    row = i // 8
    col = i %  8

    ax[row][col].imshow(x[i])

plt.show()

### Testing in dataset for object detection ###
pascal_voc_dir = '/home/hieu/Downloads/road_signs'
darknet_dir = '/home/hieu/Downloads/road_signs/images'

loader_pascal = obj_detection.DataLoader(pascal_voc_dir, format_='pascal_voc', batch_size=64)
dataset = loader_pascal.get_train_dataset()
print('[INFO] Dataset in pascal voc format loaded')
imgs, boxes, labels = next(iter(dataset))
imgs = ((imgs.numpy() * 127.5) + 127.5).astype(np.uint8)

fig, ax = plt.subplots(8,8, figsize=(20,20))
for i in range(64):
    row = i // 8
    col = i %  8
    H, W = imgs[i].shape[:2]
    x, y, w, h = (boxes[i].numpy() * np.array([W, H, W, H])).astype('int')

    img = cv2.rectangle(imgs[i], (x, y), (x+w, y+h), (0,255,0), 1)

    ax[row][col].imshow(img)

plt.show()


loader_darknet = obj_detection.DataLoader(darknet_dir, format_='darknet', batch_size=64)
dataset = loader_darknet.get_train_dataset()
print('[INFO] Dataset in darknet format loaded')
imgs, boxes, labels = next(iter(dataset))
imgs = ((imgs.numpy() * 127.5) + 127.5).astype(np.uint8)

fig, ax = plt.subplots(8,8, figsize=(20,20))
for i in range(64):
    row = i // 8
    col = i %  8
    H, W = imgs[i].shape[:2]
    x, y, w, h = (boxes[i].numpy() * np.array([W, H, W, H])).astype('int')

    img = cv2.rectangle(imgs[i], (x, y), (x+w, y+h), (0,255,0), 1)

    ax[row][col].imshow(img)

plt.show()
