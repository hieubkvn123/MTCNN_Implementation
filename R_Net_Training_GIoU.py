import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

### Other dependencies ###
from PIL import Image
from callbacks import CustomValidationCallback
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

### Tensorflow dependencies ###
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

### Some constants ###
# train_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSDB/TrainIJCNN2013/TrainIJCNN2013"
train_dir = "/home/minhhieu/Desktop/Hieu/datasets/RoadSignDetection/images"
test_dir = "/home/minhhieu/Desktop/Hieu/datasets/GTSDB/TestIJCNN2013/TestIJCNN2013Download"
rnet_weights = "weights/rnet.weights.hdf5"

validation_size = 0.2
epochs = 2500
batch_size = 8
epochs = 2500
decay_rate = 1e-8
tensorboard = TensorBoard(log_dir="./logs_rnet")
checkpoint = ModelCheckpoint(rnet_weights, save_weights_only=True, verbose=1)
early_stop_1 = EarlyStopping(monitor='val_bbox_regression_loss', patience=20, verbose=1)
early_stop_2 = EarlyStopping(monitor='val_probability_loss', patience=10, verbose=1)
lr_scheduler = LearningRateScheduler(lambda epoch, lr : lr if (epoch < 40) else lr / (1 + decay_rate * epoch), verbose=1)
callbacks = [
    tensorboard,
    checkpoint,
    early_stop_1,
    early_stop_2,
    lr_scheduler
]

# # Load and explore dataset
def load_raw_dataset(dataset_dir, gt_file, delimiter=';'):
    '''
        This function will take in a dataset directory with ppm images (according to the DTSDB dataset)
        then it will return a list where each element is a list of 3 items. First item is the image, the
        second item is the bounding box and the last is the class ID.
        
        Params :
            @dataset_dir : Dataset directory.
            @gt_file : The file that consists of ground truth annotation in the format
            <img_pth>;<left>;<top>;<right>;<bottom>;<class_idx>.
            @delimiter : The separator of each item in each line of the ground truth file
            
        Returns :
            raw_dataset : list of elements [<cv2_img>, <gt_bbox>, <class_idx>]
    '''
    gt_abs_path = os.path.join(dataset_dir, gt_file)
    lines = open(gt_abs_path, 'r').readlines()
    
    images_to_gt = [[x.strip().split(delimiter)[0],   # Image path
                     x.strip().split(delimiter)[1:5], # Bbox regression ground truth
                     x.strip().split(delimiter)[5]]   # The class index
                    for x in lines]
    
    raw_dataset = [[cv2.imread(os.path.join(dataset_dir, x[0])),
                    np.array(x[1]).astype('int'),
                    int(x[2])]
                  for x in images_to_gt]
    
    print(f'[INFO] {len(raw_dataset)} samples loaded ... ')
    
    return raw_dataset
    
raw_dataset = load_raw_dataset(train_dir, 'gt.txt')

### Visualize sample data ###
img = raw_dataset[1][0].copy()
bbox = raw_dataset[1][1]
x1, y1, x2, y2 = bbox
img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
plt.imshow(img)
plt.show()


# ### Generating negative samples (samples without traffic signs)
def generate_neg_samples(raw_dataset, crop_size=(48, 48)):
    '''
        This function will generate croppings of fixed size without any traffic sign
        and return those croppings with dummy bbox ground truth
        
        Params:
            @raw_dataset : The raw dataset formated similarly above
            @crop_size : The fixed cropping size
            
        Return:
            neg_samples : A list in format [<crop_img>,<dummy_bbox>,0]
    '''
    neg_samples = []
    for img, bbox, class_id in raw_dataset:
        height, width = img.shape[:2]
        x1, y1, x2, y2 = bbox
        range_x = ((0, max(x1 - crop_size[0], 0)), (min(x2 + crop_size[0], width), width))
        range_y = ((0, max(y1 - crop_size[1], 0)), (min(y2 + crop_size[1], height), height))
        
        ### Generate a random x,y coordinates ###
        x = random.choice([random.randint(range_x[0][0], range_x[0][1]), 
                           random.randint(range_x[1][0], range_x[1][1])])
        y = random.choice([random.randint(range_y[0][0], range_y[0][1]), 
                           random.randint(range_y[1][0], range_y[1][1])])
        
        # regenerate if cropping does not satisfy the size requirements
        while(width - x < crop_size[0] or height - y < crop_size[1]):
            x = random.choice([random.randint(range_x[0][0], range_x[0][1]), 
                           random.randint(range_x[1][0], range_x[1][1])])
            y = random.choice([random.randint(range_y[0][0], range_y[0][1]), 
                           random.randint(range_y[1][0], range_y[1][1])])
        
        crop = img[y:y+crop_size[1], x:x+crop_size[0]]
        neg_samples.append([crop, np.array([0,0,0,0]), 0])
        
    print(f'[INFO] {len(neg_samples)} negative samples generated ... ')
    return np.array(neg_samples)

neg_samples = generate_neg_samples(raw_dataset, crop_size=(48,48))
plt.imshow(neg_samples[1][0].copy())
plt.show()


# ### Generate positive samples (samles with traffic signs)
def generate_pos_samples(raw_dataset, pad_range=(10, 100), img_size=48):
    '''
        This function will generate croppings with traffic signs
        and return those croppings with bbox ground truth
        
        Params:
            @raw_dataset : The raw dataset formated similarly above
            @pad_range : The pad range around the ground truth bounding box
            
        Return:
            pos_samples : A list in format [<crop_img>,<bbox>,1]
    '''
    pos_samples = []
    for img, bbox, class_id in raw_dataset:
        height, width = img.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x1 = min(x1, random.randint(pad_range[0], pad_range[1]))
        pad_x2 = min(width - x2, random.randint(pad_range[0], pad_range[1]))
        pad_y1 = min(y1, random.randint(pad_range[0], pad_range[1]))
        pad_y2 = min(height - y2, random.randint(pad_range[0], pad_range[1]))
        
        crop = img[y1 - pad_y1:y2 + pad_y2, x1 - pad_x1:x2 + pad_x2]
        h, w = crop.shape[:2]
        
        gt = np.array([pad_x1, pad_y1, pad_x1 + x2 - x1, pad_y1 + y2 - y1])
        gt[2] = gt[2] - gt[0] # Calculates width
        gt[3] = gt[3] - gt[1] # Calculates height
        gt = np.multiply(gt, np.array([1/w, 1/h, 1/w, 1/h])).astype('float32')
        
        crop = cv2.resize(crop, (img_size, img_size))
        pos_samples.append([crop, gt, 1])
        
    print(f'[INFO] {len(pos_samples)} positive samples generated ... ')
    return np.array(pos_samples)

def get_custom_bbox_regression_loss(reduction='sum', batch_size=16):
    def custom_bbox_regression_loss(y_true, y_pred, reduction=reduction, batch_size=batch_size):
        '''
            This function customize bounding box regression loss by taking sum of the 
            l2 loss of the top left corner coordinates and the log of the ration between
            predicted and ground truth width/height.

            Params :
                - y_true : ground truth bounding boxes.
                - y_pred : predicted bounding boxes.

            Return :
                - loss : custom bbox regression loss
        '''
        def get_mask(batch_size, col_id=0):
            a = np.zeros((batch_size, 4), dtype=np.float32)
            a[:, col_id] = 1
            a = tf.convert_to_tensor(a)

            return a

        y_true = tf.reshape(y_true, [-1, 4])
        y_pred = tf.reshape(y_pred, [-1, 4])

        if(reduction == 'sum'):
            reduction = K.sum
        elif(reduction == 'mean'):
            reduction = K.mean
        else:
            reduction = K.mean

        if(isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
            x_gt, y_gt, w_gt, h_gt = y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3]
            x_pr, y_pr, w_pr, h_pr = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3]
        else:
            x_gt = K.sum(tf.multiply(y_true, get_mask(batch_size, col_id=0)), axis=1, keepdims=True) 
            y_gt = K.sum(tf.multiply(y_true, get_mask(batch_size, col_id=1)), axis=1, keepdims=True) 
            w_gt = K.sum(tf.multiply(y_true, get_mask(batch_size, col_id=2)), axis=1, keepdims=True) 
            h_gt = K.sum(tf.multiply(y_true, get_mask(batch_size, col_id=3)), axis=1, keepdims=True) 
                                                      
            x_pr = K.sum(tf.multiply(y_pred, get_mask(batch_size, col_id=0)), axis=1, keepdims=True) 
            y_pr = K.sum(tf.multiply(y_pred, get_mask(batch_size, col_id=1)), axis=1, keepdims=True) 
            w_pr = K.sum(tf.multiply(y_pred, get_mask(batch_size, col_id=2)), axis=1, keepdims=True) 
            h_pr = K.sum(tf.multiply(y_pred, get_mask(batch_size, col_id=3)), axis=1, keepdims=True) 
       
        l2_loss = reduction((x_gt - x_pr) ** 2) + reduction((y_gt - y_pr) ** 2)
        log_loss = reduction(K.binary_crossentropy(w_gt, w_pr)) + reduction(K.binary_crossentropy(h_gt, h_pr))
        # log_loss = reduction((K.sqrt(w_gt) - K.sqrt(w_pr)) ** 2) + reduction((K.sqrt(h_gt) - K.sqrt(h_pr))**2)
        loss = l2_loss + log_loss

        return loss

    return custom_bbox_regression_loss

pos_samples = generate_pos_samples(raw_dataset, pad_range=(5, 50), img_size=48)

img, gt, label = pos_samples[6]
img = img.copy()
h, w = img.shape[:2]
x1, y1, w, h = np.multiply(gt, np.array([w,h,w,h])).astype('int')

img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), 2)
plt.imshow(img)
plt.show()


# ### Combine negative and positive samples to form training dataset
# Concatenate two groups and shuffle
train_dataset = np.concatenate([pos_samples, neg_samples])
np.random.shuffle(train_dataset)

images = np.array([x[0] for x in train_dataset])
bboxes = np.array([x[1] for x in train_dataset])
labels = OneHotEncoder().fit_transform(train_dataset[:,2].reshape(-1, 1)).toarray()

bboxes = bboxes.reshape(-1, 1, 1, 4)
labels = labels.reshape(-1, 1, 1, 2)

print(images.shape)
images = ((images - 127.5) / 127.5).astype('float32')
bboxes = bboxes.astype('float32')
labels = labels.astype('float32')

dataset_size = images.shape[0]
train_size = int(dataset_size * (1 - validation_size))

train_images = images[:train_size]
train_bboxes = bboxes[:train_size]
train_labels = labels[:train_size]

val_images = images[train_size:]
val_bboxes = bboxes[train_size:]
val_labels = labels[train_size:]

# # Implement P-Net architecture
def build_rnet_model(batch_norm=True, dropout=False):
    inputs = Input(shape=(48, 48, 3))
    
    r_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(inputs)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    r_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(r_layer)

    r_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    
    r_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)
    r_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(r_layer)
    
    r_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(r_layer)
    r_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(batch_norm) : r_layer = BatchNormalization()(r_layer)

    r_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(r_layer)
    r_layer = PReLU(shared_axes=[1, 2])(r_layer)
    if(dropout) : r_layer = Dropout(0.5)(r_layer)

    r_layer_out1 = Conv2D(32, kernel_size=(1,1), strides=(1,1))(r_layer)
    r_layer_out1 = Conv2D(2, kernel_size=(1,1), strides=(1,1))(r_layer_out1)
    r_layer_out1 = Softmax(axis=3, name='probability')(r_layer_out1)

    r_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', name='bbox_regression')(r_layer)

    r_net = Model(inputs, [r_layer_out1, r_layer_out2], name='R-Net')

    return r_net

rnet = build_rnet_model(batch_norm=True, dropout=True)
print(rnet.summary())


# # Start training
losses_1 = {
    'probability' : BinaryCrossentropy(),
    'bbox_regression' : get_custom_bbox_regression_loss(batch_size=batch_size, reduction='mean')
}

losses_2 = {
    'probability' : BinaryCrossentropy(),
    'bbox_regression' : get_custom_bbox_regression_loss(batch_size=batch_size, reduction='mean') 
}

loss_weights = {
    'probability' : 1.0,
    'bbox_regression' : 0.5
}

y_train = {
    'probability' : train_labels,
    'bbox_regression' : train_bboxes
}

y_val = {
    'probability' : val_labels,
    'bbox_regression' : val_bboxes
}

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, y_train))
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, y_val))
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

if(os.path.exists(rnet_weights)):
    print(f'[INFO] Loading pretrained weights from {rnet_weights}')
    rnet.load_weights(rnet_weights)

### Schedule 1 ###
print('[INFO] Begining schedule #1 ... ')
rnet.compile(optimizer=Adam(lr=0.00001, amsgrad=True),
            loss=losses_1,
            loss_weights=loss_weights,
            metrics={'probability':'accuracy'})

history = rnet.fit(train_dataset, epochs=epochs//2, 
        batch_size=batch_size, 
        validation_data=val_dataset, validation_batch_size=batch_size, 
        callbacks=callbacks)

### Schedule  2 ###
print('[INFO] Begining schedule #2 ...')
rnet.compile(optimizer=Adam(lr=0.0000001, amsgrad=True),
            loss=losses_2,
            loss_weights=loss_weights,
            metrics={'probability':'accuracy'})

history = rnet.fit(train_dataset, initial_epoch=epochs//2,
        epochs=epochs, batch_size=batch_size, 
        validation_data=val_dataset, validation_batch_size=batch_size, 
        callbacks=callbacks)

