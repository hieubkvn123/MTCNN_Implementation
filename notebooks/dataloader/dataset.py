import os
import tqdm
import glob
import numpy as np
import tensorflow as tf

from dataset_ import *
from tfrecord import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, directory, labels=None, one_hot=False,
                 labels_as_subdir=False,
                 batch_size = 32,
                 img_size=64,
                 train_val_ratio=0.333,
                 preprocess="default",
                 color_space="rgb",
                 random_noise=False,
                 drop_remainder=True,
                 shuffle=True,
                 repeat = 1):
        ''''''

        rgb_to_rgb = lambda img : img
        extensions = ['*.png', "*.jpg", "*.jpeg"]
        valid_preprocess = ["default", "standard", "minmax"]
        valid_colors = {
            "rgb" : rgb_to_rgb,
            "hsv" : tf.image.rgb_to_hsv,
            "gray" : tf.image.rgb_to_grayscale
        }
        img_paths = []

        for ext in extensions:
            img_paths += glob.glob(os.path.join(directory, ext))

        if(labels_as_subdir and labels is None):
            labels = np.array([x.split('/')[-2] for x in img_paths])

        ### Make sure user inputs are valid ###
        if (preprocess not in valid_preprocess):
            raise Exception(f'Invalid preprocessing method, methods include {valid_preprocess}')
        if(color_space not in valid_colors):
            raise Exception(f'Invalid color space, valid color spaces are {list(valid_colors.keys())}')
        if(labels is not None):
            if (not (isinstance(labels, np.ndarray) or isinstance(labels, list))):
                raise Exception('If labels are specified, it must be a numpy array or a list')
            if(len(img_paths) != len(labels)):
                raise Exception('Labels and images array length not consistent')

            labels = np.array(labels)
            if(labels.dtype != np.int32):
                labels = LabelEncoder().fit_transform(labels)
                labels = labels.astype(np.int32)

        if(img_paths == []):
            raise Exception('There are no images in given directory ...')

        ### Start making the dataset ###
        self.img_size = img_size
        self.batch_size = batch_size
        self.random_noise = random_noise
        self.colorspace = valid_colors[color_space]
        if(preprocess == "standard"):
            self.map_fn = self.get_image_standardization_map_fn(self.img_size)
        elif(preprocess == "minmax"):
            self.map_fn = self.get_minmax_map_fn(self.img_size)
        else:
            self.map_fn = self.get_default_map_fn(self.img_size)

        train_labels = None
        val_labels = None
        train_img_paths = []
        val_img_paths = []

        if(labels is not None):
            train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(img_paths, labels, test_size=train_val_ratio)
        else:
            train_img_paths, val_img_paths = train_test_split(img_paths, test_size=train_val_ratio)

        if(one_hot and labels is not None):
            train_labels = tf.one_hot(train_labels, depth=len(np.unique(labels)))
            val_labels   = tf.one_hot(val_labels, depth=len(np.unique(labels)))

        self.img_paths = img_paths
        self.labels = labels
        self.train_dataset = disk_image_batch_dataset(train_img_paths,
                                          self.batch_size,
                                          labels = train_labels,
                                          drop_remainder=drop_remainder,
                                          map_fn=self.map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
        
        self.val_dataset = disk_image_batch_dataset(val_img_paths,
                                          self.batch_size,
                                          labels = val_labels,
                                          drop_remainder=drop_remainder,
                                          map_fn=self.map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)

        self.img_shape = (img_size, img_size, 3)
        self.dataset_len = len(img_paths) // self.batch_size

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def add_random_noise(self, img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_saturation(img, 0.6, 1.4)
        img = tf.image.random_brightness(img, 0.4)

        return img

    def get_default_map_fn(self, img_size):
        @tf.function
        def map_fn(img, random_noise=self.random_noise, noise_fn=self.add_random_noise):
            img = tf.image.resize(img, [img_size, img_size])
            img = self.colorspace(img)
            if(random_noise): img = noise_fn(img)
            img = tf.clip_by_value(img, 0, 255)

            img = img / 127.5 - 1

            return img

        return map_fn

    def get_image_standardization_map_fn(self, img_size):
        @tf.function
        def map_fn(img, random_noise=self.random_noise, noise_fn=self.add_random_noise):
            img = tf.image.resize(img, [img_size, img_size])
            img = self.colorspace(img)
            if(random_noise): img = noise_fn(img)
            img = tf.clip_by_value(img, 0, 255)

            img = tf.image.per_image_standardization(img)

            return img

        return map_fn

    def get_minmax_map_fn(self, img_size, epsilon=1e-10):
        @tf.function
        def map_fn(img, random_noise=self.random_noise, noise_fn=self.add_random_noise):
            img = tf.image.resize(img, [img_size, img_size])
            img = self.colorspace(img)
            if(random_noise): img = noise_fn(img)
            img = tf.clip_by_value(img, 0, 255)

            img = tf.cast(img, tf.float32)
            min_val = tf.reduce_min(img)
            max_val = tf.reduce_max(img)
            img = (img - min_val) / tf.maximum(max_val - min_val, epsilon)

            return img

        return map_fn

    def parse_to_tfrecord(self, output_file): 
        if(os.path.exists(output_file)):
            print('[INFO] Removing old tfrecord file...')
            os.remove(output_file)

        with tqdm.tqdm(total=len(self.img_paths)) as pbar:
            with tf.io.TFRecordWriter(output_file) as writer:
                if(self.labels is not None):
                    for img_path, label in zip(self.img_paths,  self.labels):
                        tf_example = make_example(str.encode(img_path), class_id=label)
                        writer.write(tf_example.SerializeToString())
                        pbar.update(1)
                else:
                    for img_path in self.img_paths:
                        tf_example = make_example(str.encode(img_path))
                        writer.write(tf_example.SerializeToString())
                        pbar.update(1)

        print(f'[INFO] {len(self.img_paths)} samples written to {output_file}')
