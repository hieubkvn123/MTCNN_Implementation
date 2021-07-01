import os
import cv2
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class MTCNN:
    def __init__(self, weights_dir):
        self.pnet_weights = os.path.join(weights_dir, 'pnet.weights.hdf5')
        self.rnet_weights = os.path.join(weights_dir, 'rnet.weights.hdf5')
        self.onet_weights = os.path.join(weights_dir, 'onet.weights.hdf5')

        self.pnet_configs = json.load(open(os.path.join(weights_dir, 'pnet.json'), 'r'))
        self.rnet_configs = json.load(open(os.path.join(weights_dir, 'rnet.json'), 'r'))
        self.onet_configs = json.load(open(os.path.join(weights_dir, 'onet.json'), 'r'))

        ### Construct network architectures ###
        self.pnet = self._build_pnet_model(input_shape=self.pnet_configs['input_shape'], 
                                          batch_norm=self.pnet_configs['batch_norm'], 
                                          dropout=self.pnet_configs['dropout'],
                                          n_classes=self.pnet_configs['n_classes'])

        self.rnet = self._build_pnet_model(input_shape=self.rnet_configs['input_shape'], 
                                          batch_norm=self.rnet_configs['batch_norm'], 
                                          dropout=self.rnet_configs['dropout'],
                                          n_classes=self.rnet_configs['n_classes'])
    

        self.onet = self._build_pnet_model(input_shape=self.onet_configs['input_shape'], 
                                          batch_norm=self.onet_configs['batch_norm'], 
                                          dropout=self.onet_configs['dropout'],
                                          n_classes=self.onet_configs['n_classes'])
        
        ### Load weights ###
        self.pnet.load_weights(self.pnet_weights)
        self.rnet.load_weights(self.rnet_weights)
        self.onet.load_weights(self.onet_weights)

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

    def __nms(self, boxes, s, threshold, method):
        """
            Non Maximum Suppression.

            Params:
                @param boxes: np array with bounding boxes.
                @param threshold:
                @param method: NMS method to apply. Available values ('Min', 'Union')

            Return:
                pick : An array of indices selected.
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2] + x1
        y2 = boxes[:, 3] + y1

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if method == 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    def stage1_pnet(self, raw_img, threshold=0.5, nms_threshold=0.5,
                    scale_factor=2.0, min_img_size = 48, padding = 0.15, visualize=False):
        '''

        '''
        H, W = raw_img.shape[:2]
        images = [raw_img]
        current_h, current_w = raw_img.shape[:2]

        ### 1. Get image pyramid ###
        while(current_h > min_img_size and current_w > min_img_size):
            current_h = int(current_h / scale_factor)
            current_w = int(current_w / scale_factor)

            if(current_w < min_img_size or current_h < min_img_size) : break

            image = cv2.resize(raw_img, (current_w, current_h))
            images.append(image)

        ### 2. Get bounding boxes from each image in the pyramid ###
        boxes = []
        for i, image in enumerate(images):
            if(i == 0): scale = 1
            else : scale = scale_factor ** i

            img = (image - 127.5) / 127.5
            height, width = image.shape[:2]

            predictions = self.pnet.predict(np.array([img]))
            bboxes = predictions[1][0]
            confidence = predictions[0][0]

            ### Getting confidence map ###
            conf_map = np.max(confidence[:, :, :], axis=2)
            conf_map[conf_map > threshold] = 1.0
            conf_map[conf_map <= threshold] = 0
            conf_map = (conf_map * 255).astype(np.uint8)
            contours, hierarchy = cv2.findContours(conf_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                rect = cv2.boundingRect(contour)
                x, y, w, h = (rect * np.array([W/conf_map.shape[1],H/conf_map.shape[0],W/conf_map.shape[1],H/conf_map.shape[0] ])).astype(int)

                x -= min(int(padding * w), x)
                y -= min(int(padding * h), y)
                w += 2*int(padding * w)
                h += 2*int(padding * h)
                if(w * h < (W * H)/64): continue
                boxes.append([x,y,w,h])

        ### 3. Performing nms ###
        crops = []
        # pick = self.__nms(np.array(boxes), np.ones((len(boxes))), nms_threshold, 'Min')
        for i in range(len(boxes)):
            (x, y, w, h) = boxes[i]
            crops.append([(x,y,w,h), raw_img[y:y+h, x:x+w]])
        
        return crops

    def stage2_rnet(self, raw_img, crops):
        boxes = []
        for i in crops:
            crop = i[1]
            (x, y, w, h) = i[0]
            H, W = crop.shape[:2]
            img = cv2.resize(crop, (self.rnet_configs['input_shape'], self.rnet_configs['input_shape']))
            img = (img - 127.5) / 127.5

            prediction = self.rnet.predict(np.array([img]))
            confidence = prediction[0][0][0][0]
            confidence = confidence[np.argmax(confidence)]

            bbox = prediction[1][0][0][0]
            if(confidence < 0.8) : continue

            x_,y_,w,h = (bbox * np.array([W, H, W, H])).astype('int')
            x+=x_
            y+=y_

            boxes.append([x, y, w, h])

        crops = []
        pick = self.__nms(np.array(boxes), np.ones((len(boxes))), 0.3, 'Max')
        for i in pick:
            (x, y, w, h) = boxes[i]
            crops.append([(x,y,w,h), raw_img[y:y+h, x:x+w]])

        return crops

    def stage3_onet(self, crops):
        boxes = []
        labels = []
        for i in crops:
            crop = i[1]
            (x, y, w, h) = i[0]
            H, W = crop.shape[:2]
            img = cv2.resize(crop, (self.onet_configs['input_shape'], self.onet_configs['input_shape']))
            img = (img - 127.5) / 127.5

            prediction = self.onet.predict(np.array([img]))
            confidence = prediction[0][0][0][0]
            confidence = confidence[np.argmax(confidence)]
            label = np.argmax(confidence)

            bbox = prediction[1][0][0][0]
            if(confidence < 0.98) : continue

            x_,y_,w,h = (bbox * np.array([W, H, W, H])).astype('int')
            x+=x_
            y+=y_

            boxes.append([x, y, w, h])
            labels.append(label)

        return boxes, labels

    def predict(self, img):
        crops = self.stage1_pnet(img)
        crops = self.stage2_rnet(img, crops)
        boxes, labels = self.stage3_onet(crops)

        return boxes, labels
    
mtcnn = MTCNN('weights/road_signs_w_dataloader')
img = cv2.imread('test/test1.jpg')
boxes, labels = mtcnn.predict(img)
print(boxes, labels)
