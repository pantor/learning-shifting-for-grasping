import imageio
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import backend as K


def load_data(path, label_fields=['did_grasp', 'gripper_class']):
    data = pd.read_csv(path)
    labels = data[label_fields].values
    image_paths = data['image_path'].values
    images = np.array([np.expand_dims(imageio.imread(path, pilmode='P'), axis=2) for path in image_paths]) / 255.
    return images, labels

def single_class_split(y_true, y_pred):
    value_true = y_true[:, 0]
    index = tf.to_int32(y_true[:, 1])
    indices_gripper_class = tf.stack([tf.range(tf.shape(index)[0]), index], axis=1)
    value_pred = tf.gather_nd(y_pred, indices_gripper_class)
    return value_true, value_pred

def crossentropy(y_true, y_pred):
    value_true, value_pred = single_class_split(y_true, y_pred)
    return k.metrics.binary_crossentropy(value_true, value_pred)

def accuracy(y_true, y_pred):
    value_true, value_pred = single_class_split(y_true, y_pred)
    return K.mean(tf.to_float(K.equal(K.round(value_pred), value_true)))

def precision(y_true, y_pred):
    value_true, value_pred = single_class_split(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(value_true * value_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(value_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())