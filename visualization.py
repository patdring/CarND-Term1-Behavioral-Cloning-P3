import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import tensorflow
import keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, ELU, Activation
from keras.layers import Lambda, Dropout, Cropping2D, SpatialDropout2D

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras import backend as K
from keras.utils import np_utils

def create_model():
    model = Sequential()

    # trim image to only see section with road
    model.add(Cropping2D(cropping=((50,20), (0,10)), input_shape=(160,320,3)))

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # Nvidia model
    model.add(Conv2D(24, (5, 5), activation="relu", name="conv_1", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", name="conv_3", strides=(2, 2)))
    model.add(SpatialDropout2D(.5, dim_ordering='default'))
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    return model


model = create_model()
model.load_weights('model.h5')

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./pics/nvidia_model.png', show_shapes=True, show_layer_names=True)

images = []
image = cv2.imread('./pics/center_2016_12_01_13_42_42_686.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append(image)

image = cv2.imread('./pics/right_2016_12_01_13_32_52_652.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append(image)

image = cv2.imread('./pics/center_2016_12_01_13_40_11_279.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append(image)

from kerastoolbox.visu import plot_feature_map

x = plot_feature_map(model, X=images, layer_id=2, n_columns=3, n=256)
x.savefig('./pics/conv_1.png')
x = plot_feature_map(model, X=images, layer_id=3, n_columns=3, n=256)
x.savefig('./pics/conv_2.png')
x = plot_feature_map(model, X=images, layer_id=4, n_columns=3, n=256)
x.savefig('./pics/conv_3.png')
x = plot_feature_map(model, X=images, layer_id=6, n_columns=3, n=256)
x.savefig('./pics/conv_4.png')
x = plot_feature_map(model, X=images, layer_id=7, n_columns=3, n=256)
x.savefig('./pics/conv_5.png')

