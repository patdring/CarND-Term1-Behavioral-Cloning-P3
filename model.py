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

def load_driving_log(file):
    lines = []
    with open(file) as logfile:
        reader = csv.reader(logfile)
        for line in reader:
            lines.append(line)
    # skip first line/header
    return lines[1:]

def load_driving_data(log_path, data_path):
    path = data_path
    lines = log_path

    shuffle(lines)

    images = []
    angles = []

    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            curr_path = path+filename
            image = cv2.imread(curr_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            angle = float(line[3])

            if i == 0:
                angles.append(angle)

            if i == 1:
                angles.append(angle + 0.25)

            if i == 2:
                angles.append(angle - 0.25)

    augmented_images = []
    augmented_angles = []

    for (image, angle) in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image, 1))
        augmented_angles.append(angle * -1.0)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_angles)

    return shuffle(X_train, y_train)

driving_log = load_driving_log('../data/driving_log.csv')

train_samples, valid_samples = train_test_split(driving_log, shuffle=True, test_size=0.2)

X_train, y_train = load_driving_data(train_samples, '../data/IMG/')
X_valid, y_valid = load_driving_data(valid_samples, '../data/IMG/')

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

model.compile(loss='mse', optimizer='adam')

model.summary()

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=2, epochs=7, verbose=1)

model.save('model.h5')
