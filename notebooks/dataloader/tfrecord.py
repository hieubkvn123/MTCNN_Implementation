import numpy as np
import tensorflow as tf

from dataset_ import batch_dataset

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_example(img_path, class_id=None):
  feature = {
    'image/path' : _bytes_feature(img_path)
  }
  if(class_id is not None):
    feature = {
      'image/path' : _bytes_feature(img_path),
      'image/class' : _int64_feature(class_id)
    }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def _parse_tfrecord(label=False, n_classes=None):
  def parse_tfrecord(tfrecord):
      ''' Each record will have image path and class name in int64 '''
  
      if(label):
        features = {
            'image/path' : tf.io.FixedLenFeature([], tf.string),
            'image/class': tf.io.FixedLenFeature([], tf.int64)
        }
      else:
        features = {
          'image/path' : tf.io.FixedLenFeature([], tf.string)
        }

      example =  tf.io.parse_single_example(tfrecord, features)
      img = tf.io.read_file(example['image/path'])
      img = tf.image.decode_png(img, channels=3)

      if(label):
        label = example['image/class']
        if(n_classes is not None):
          label = tf.one_hot(label, depth=n_classes)
        
        return img, label
      else:
        return img
  return parse_tfrecord

def tfrecord_image_batch_dataset(tfrecord_file,
                              batch_size,
                              label=False,
                              n_classes=None,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset.repeat()
    dataset.map(_parse_tfrecord(label=label,n_classes=n_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset

