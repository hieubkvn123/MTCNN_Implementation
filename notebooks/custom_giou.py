import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K
from tensorflow.keras.metrics import kl_divergence

### GIoU formula ###
def GIoU(mode='giou', reg_factor=None):
    def get_giou_loss(b1, b2):
        b2 = tf.cast(b2, dtype=tf.float32)
        b1 = tf.cast(b1, dtype=tf.float32)
        zero = tf.convert_to_tensor(0.0, dtype=b1.dtype)
        reg = 0

        b1_xmin, b1_ymin, b1_width, b1_height = tf.unstack(b1, 4, axis=-1)
        b2_xmin, b2_ymin, b2_width, b2_height = tf.unstack(b2, 4, axis=-1)
        
        if(reg_factor is not None):
            reg = reg_factor * (kl_divergence(b1_width, b2_width) + kl_divergence(b1_height, b2_height))
            reg = tf.reduce_mean(reg)

        b1_xmax = tf.cast(b1_xmin + b1_width, dtype=tf.float32)
        b1_ymax = tf.cast(b1_ymin + b1_height, dtype=tf.float32)
        b2_xmax = tf.cast(b2_xmin + b2_width, dtype=tf.float32)
        b2_ymax = tf.cast(b2_ymin + b2_height, dtype=tf.float32)

        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height

        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        if mode == "iou":
            return 1 - tf.math.reduce_mean(iou) + reg

        enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
        enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
        enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
        enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
        enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height
        giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)

        return 1 - K.mean(giou) + reg
    
    return get_giou_loss
'''
# Bounding box in darknet format - x, y, w, h
box1 = tf.constant([[[1,2,2,2]]], dtype=tf.float64)
box2 = tf.constant([[[2,3,2,2]]], dtype=tf.float64)

# Bounding box in tfa format - y1, x1, y2, x2
box1_ = tf.constant([[[2,1,4,3]]], dtype=tf.float64)
box2_ = tf.constant([[[3,2,5,4]]], dtype=tf.float64)

### Testing ###
print(GIoU(mode='iou', reg_factor=None)(box1, box2))
print(tfa.losses.GIoULoss(mode='iou')(box1_, box2_))

print(GIoU(mode='giou', reg_factor=None)(box1, box2))
print(tfa.losses.GIoULoss(mode='giou')(box1_, box2_))

### Testing with regularization ###
box1 = tf.constant([[[1,2,2,2]]], dtype=tf.float64)
box2 = tf.constant([[[2,3,4,2]]], dtype=tf.float64)
print(GIoU(mode='iou', reg_factor=2e-4)(box1, box2))
'''
