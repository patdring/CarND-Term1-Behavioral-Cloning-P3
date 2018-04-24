import csv
import cv2
import numpy as np
import sklearn
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread('./data/IMG/'+batch_sample[0].split('/')[-1])
                if center_image is not None:              
                    center_image_YUV = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                    center_angle = float(batch_sample[3])
                    images.append(center_image_YUV)
                    angles.append(center_angle)
                
                left_image = cv2.imread('./data/IMG/'+batch_sample[1].split('/')[-1])
                if left_image is not None:
                    left_image_YUV = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                    left_angle = float(batch_sample[3])
                    images.append(left_image_YUV)
                    angles.append(center_angle + 0.2)

                right_image = cv2.imread('./data/IMG/'+batch_sample[2].split('/')[-1])
                if right_image is not None:
                    right_image_YUV = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                    right_angle = float(batch_sample[3])
                    images.append(right_image_YUV)
                    angles.append(center_angle - 0.2)

            if len(images) == 0:
                continue

            augmented_images = []
            augmented_angles = []
            for (image, angle) in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)
    
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)

batch = 1

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
model.add(Dropout(.2))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
model.add(Dropout(.2))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
model.add(Dropout(.2))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(.2))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Flatten())
model.add(Dense(1000, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

print('Training the model')

model.compile(optimizer='adam', loss='mse')

model.fit_generator(
    generator=generator(train_samples, batch_size=batch),
    steps_per_epoch=len(train_samples)*6,
    epochs=3,
    validation_data=generator(validation_samples, batch_size=batch),
    validation_steps=len(validation_samples)
)

print('Saved the model')
model.save('model.h5')
