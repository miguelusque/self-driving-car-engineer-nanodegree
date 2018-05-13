import csv
import cv2
import numpy as np
import tensorflow as tf

# My humble tribute to Michael Jordan and Magic Johnson, 
# the best basketball players ever. 
np.random.seed(23)
tf.set_random_seed(32)

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.3
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        if i == 1:
            measurement = measurement + (0.9 * correction)
        elif i == 2:
            measurement = measurement - (1.1 * correction)
    
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip (images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print (X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers


model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')