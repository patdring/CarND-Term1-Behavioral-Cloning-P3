import os
import csv

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread('../data/IMG/'+batch_sample[0].split('/')[-1])  
                left_image = cv2.imread('../data/IMG/'+batch_sample[1].split('/')[-1]) 
                right_image = cv2.imread('../data/IMG/'+batch_sample[2].split('/')[-1])

                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)

                center_angle = float(batch_sample[3])

                images.extend([center_image, left_image, right_image])
                angles.extend([
                    center_angle,
                    center_angle + 0.2,
                    center_angle - 0.2
                ])

            augmented_images = []
            augmented_angles = []
            for (image, angle) in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='elu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

history_object = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples)*6,
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=5)

model.save("model.h5")

import matplotlib.pyplot as plt

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

