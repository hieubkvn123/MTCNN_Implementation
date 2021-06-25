import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class CustomValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, bbox_loss, prob_loss, validation_steps=10):
        self.dataset = dataset
        self.bbox_loss = bbox_loss
        self.prob_loss = prob_loss
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        losses = []

        bbox_model = Model(inputs=self.model.inputs, outputs=self.model.get_layer('bbox_regression').output)
        prob_model = Model(inputs=self.model.inputs, outputs=self.model.get_layer('probability').output)
       
        print('\n[INFO] Start validation ...')
        for i in range(self.validation_steps):
            val_images, val_y = next(iter(self.dataset))
            print(val_y)
            val_bboxes = val_y['bbox_regression']
            val_labels = val_y['probability']
            
            predicted_bbox = bbox_model.predict(val_images)
            predicted_prob = prob_model.predict(val_images)

            bbox_loss = self.bbox_loss(val_bboxes, predicted_bbox)
            prob_loss = self.prob_loss(val_labels, predicted_prob)

            loss = bbox_loss + prob_loss
            losses.append(loss)

        print(f'\n[INFO] Validation loss = {np.array(losses).mean()}')
