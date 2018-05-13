import csv
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# My humble tribute to Michael Jordan and Magic Johnson, 
# the best basketball players ever. 
np.random.seed(23)
tf.set_random_seed(32)

# Batch_size must be a multiple of 6 [6, 12, 18, 24...]
def generator(samples, batch_size=36):
    num_samples = len(samples)

    if (batch_size > num_samples):
        batch_size = num_samples

    batch_size = int(batch_size/6)

    print(batch_size)
    correction = 0.3
    while 1: # Loop forever so the gerator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            measurements = []
            for sample in batch_samples:
                for i in range(3):
                    source_path = sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)

                    # Left correction and right correction are different
                    # because I have noticed that the car tends to one
                    # side more than the other. This is done to compensate
                    # that behaviour. That is probably due to the training
                    # data used.
                    measurement = float(sample[3])
                    if i == 1:
                        measurement = measurement + (0.9 * correction)
                    elif i == 2:
                        measurement = measurement - (1.1 * correction)
                
                    measurements.append(measurement)

            # Flipping images and ateering measurements
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip (images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

# Model design
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
model.add(Dropout(0.1))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))


# Read the driving log
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for sample in reader:
        samples.append(sample)


# Split the samples in train and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Train and valid the model using the generator function
BATCH_SIZE = 36
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# Compile and train the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
  validation_data=validation_generator, nb_val_samples=len(validation_samples), epochs=2)

# Save the model
model.save('model.h5')