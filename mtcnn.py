import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model

class MTCNN:
    def __init__(self, weights_dir):
        self.pnet_weights = os.path.join(weights_dir, 'pnet.weights.hdf5')
        self.rnet_weights = os.path.join(weights_dir, 'rnet.weights.hdf5')
        self.onet_weights = os.path.join(weights_dir, 'onet.weights.hdf5')

        self.pnet = self.build_pnet_model(input_shape=input_dim, batch_norm=True, dropout=True,
                        n_classes=n_classes)

    ### Implement the P-Net architecture ###
    def _conv_block(self, in_filters, out_filters, kernel_size=3, batch_norm=False):
        inputs = Input(shape=(None, None, in_filters))
        p_layer = Conv2D(out_filters, kernel_size=kernel_size, strides=(1, 1), padding="valid", kernel_regularizer=l2(2e-4))(inputs)
        if(batch_norm) : p_layer = BatchNormalization()(p_layer)
        p_layer = PReLU(shared_axes=[1, 2])(p_layer)

        p_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)

        block = Model(inputs = inputs, outputs=p_layer)
        return block

    def _build_pnet_model(self, input_shape=None, batch_norm=True, dropout=False, n_classes=2, activation='relu'):
        if(input_shape is not None):
            if(input_shape not in [12, 24, 48, 112, 224]):
                raise Exception('Input shape must be in 12, 24, 48')

        inputs = Input(shape=(None, None, 3))
        p_layer = self._conv_block(3, 10, kernel_size=3, batch_norm=batch_norm)(inputs)

        if(input_shape is not None):
            if(input_shape >= 24):
                p_layer = self._conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

        if(input_shape is not None):
            if(input_shape >= 48):
                p_layer = self._conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

        if(input_shape is not None):
            if(input_shape >= 112):
                p_layer = self._conv_block(10, 10, kernel_size=3, batch_norm=batch_norm)(p_layer)

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
