import multiprocessing
import numpy as np
import tensorflow as tf

from .obj_detection_utils import parse_pascal_voc, parse_darknet

def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)
    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
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


def disk_image_batch_dataset(directory,
                             format_,
                             batch_size,
                             img_dir=None,
                             annot_dir=None,
                             crop_to_bounding_box=False,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    """
    if(img_dir is None): img_dir = 'images'
    if(annot_dir is None): annot_dir = 'annotations'

    if(format_ == 'pascal_voc'):
        img_paths, bboxes, labels = parse_pascal_voc(directory, img_dir=img_dir, annot_dir=annot_dir)
    elif(format_ == 'darknet'):
        img_paths, bboxes, labels = parse_darknet(directory)

    if(len(img_paths) < 1):
        raise Exception(f"There is no image in {os.path.join(directory, img_dir)}")

    memory_data = (img_paths, bboxes, labels)
    n_classes = len(np.unique(labels))
    dataset_len = len(img_paths) // batch_size

    @tf.function
    def parse_fn(path, bbox, label):
        tf.executing_eagerly()

        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)  # fix channels to 3
        img = map_fn(img)

        ### Crop to bounding box ###
        if(crop_to_bounding_box):
            padding_range = [0, 15]
            H, W = img.shape[0], img.shape[1]
            box = bbox * tf.constant([W, H, W, H], dtype=tf.float64)
            box = tf.cast(box, dtype=tf.int64)

            pad_x = tf.math.minimum(box[0], tf.cast(abs(np.random.normal(loc=0.8,scale=0.1) * tf.cast(box[2], dtype=tf.float64)), dtype=tf.int64))#np.random.randint(padding_range[0], padding_range[1])
            pad_y = tf.math.minimum(box[1], tf.cast(abs(np.random.normal(loc=0.8,scale=0.1) * tf.cast(box[3], dtype=tf.float64)), dtype=tf.int64))#np.random.randint(padding_range[0], padding_range[1])
            pad_w = tf.math.minimum(W - (box[2] + box[0]), tf.cast(abs(np.random.normal(loc=0.8,scale=0.1) * tf.cast(box[2], dtype=tf.float64)), dtype=tf.int64))#np.random.randint(padding_range[0], padding_range[1])
            pad_h = tf.math.minimum(H - (box[3] + box[1]), tf.cast(abs(np.random.normal(loc=0.8,scale=0.1) * tf.cast(box[3], dtype=tf.float64)), dtype=tf.int64))#np.random.randint(padding_range[0], padding_range[1])

            new_x = box[0] - pad_x
            new_y = box[1] - pad_y
            new_w = box[2] + pad_x + pad_w
            new_h = box[3] + pad_y + pad_h
 
            img = tf.image.crop_to_bounding_box(img, 
                new_y, # y
                new_x, # x
                new_h, # h
                new_w  # w
            )
            bbox = tf.stack([
                pad_x / new_w,
                pad_y / new_h,
                box[2] / new_w,
                box[3] / new_h,
            ])
            
            img = tf.image.resize(img, [H, W])

        ### End of crop to bounding box ###

        return img, (bbox, label)

    def map_fn_(path, bbox, label):
        return parse_fn(path, bbox, label)# map_fn(*parse_fn_with_label(*args))

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)

    return n_classes, dataset_len, dataset

