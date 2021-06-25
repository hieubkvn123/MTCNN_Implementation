import os
import tqdm
import glob
import numpy as np
import tensorflow as tf

from obj_detection_ import *
from tfrecord import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
	def __init__(self, directory, format_,
			img_dir=None, annot_dir=None,
			one_hot=False,
			batch_size = 32,
			img_size=64,
			crop_to_bounding_box=False,
			train_val_ratio=0.333,
			preprocess="default",
			color_space="rgb",
			random_noise=False,
			drop_remainder=True,
			shuffle=True,
			repeat = 1):
		rgb_to_rgb = lambda img : img
		extensions = ['*.png', "*.jpg", "*.jpeg"]
		valid_preprocess = ["default", "standard", "minmax"]
		valid_colors = {
			"rgb" : rgb_to_rgb,
			"hsv" : tf.image.rgb_to_hsv,
			"gray" : tf.image.rgb_to_grayscale
		}
		img_paths = []

		### Configure images directory and annotation directory ###
		if(format_ not in ['pascal_voc', 'darknet']): raise Exception('Format must be pascal_voc or darknet')
		if(img_dir is None) : img_dir = 'images'
		if(annot_dir is None) : annot_dir = 'annotations'

		### Make sure user inputs are valid ###
		if (preprocess not in valid_preprocess):
			raise Exception(f'Invalid preprocessing method, methods include {valid_preprocess}')
		if(color_space not in valid_colors):
			raise Exception(f'Invalid color space, valid color spaces are {list(valid_colors.keys())}')
		
		### Start making the dataset ###
		self.img_size = img_size
		self.format_ = format_
		self.batch_size = batch_size
		self.random_noise = random_noise
		self.colorspace = valid_colors[color_space]
		if(preprocess == "standard"):
			self.map_fn = self.get_image_standardization_map_fn(self.img_size)
		elif(preprocess == "minmax"):
			self.map_fn = self.get_minmax_map_fn(self.img_size)
		else:
			self.map_fn = self.get_default_map_fn(self.img_size)

		self.train_dataset = disk_image_batch_dataset(directory,
										  self.format_,
										  self.batch_size,
										  img_dir=img_dir,
										  annot_dir=annot_dir,
										  crop_to_bounding_box=crop_to_bounding_box,
										  drop_remainder=drop_remainder,
										  map_fn=self.map_fn,
										  shuffle=shuffle,
										  repeat=repeat)
		
		self.val_dataset = disk_image_batch_dataset(directory,
										  self.format_,
										  self.batch_size,
										  img_dir=img_dir,
										  annot_dir=annot_dir,
										  crop_to_bounding_box=crop_to_bounding_box,
										  drop_remainder=drop_remainder,
										  map_fn=self.map_fn,
										  shuffle=shuffle,
										  repeat=repeat)

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