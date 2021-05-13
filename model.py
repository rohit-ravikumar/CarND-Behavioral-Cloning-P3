import csv
import cv2
from scipy import ndimage
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout,Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn

def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    while 1: 
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []
            for filepath, measurement in batch_samples:
                image = ndimage.imread(filepath)
                images.append(image)
                steering.append(measurement)
                images.append(np.fliplr(image))
                steering.append(measurement*-1.0)
            inputs = np.array(images)
            steerings = np.array(steering)
            yield sklearn.utils.shuffle(inputs, steerings)

### Read Images and save them in an array
path = '/home/workspace/More_Data_1/driving_log.csv'
lines = []
filenames = []
#images = []
measurements = []
#augmented_images = []
#augmented_measurements = []
correction_factor = 0.1
batch_size = 32
with open (path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("Number of Lines:",len(lines))

for line in lines:
    for i in range (3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        image_path = '/home/workspace/More_Data_1/IMG/'+ filename
        filenames.append(image_path)
        #image = ndimage.imread(image_path)
        #images.append(image)
        if i ==0:
            measurement = float(line[3])
            measurements.append(measurement)
        if i ==1:
            measurement = float(line[3]) + correction_factor
            measurements.append(measurement)
        if i ==2:
            measurement = float(line[3]) - correction_factor
            measurements.append(measurement)
#print("i=",i)
print("Number of Starting images",len(filenames))
print("Number of Starting measurements",len(measurements))
#print("Image Dimensions:",image.shape)

"""
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement*-1)

print("Number of Augmented images",len(augmented_images))
print("Number of Augmented measurements",len(augmented_measurements))
print("Image Dimensions:",image.shape)
"""

#samples = list(zip(augmented_images, augmented_measurements))
samples = list(zip(filenames, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Training samples:',len(train_samples))
print('Validation samples:',len(validation_samples))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model.add( Lambda( lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3) ) )
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add( Conv2D(24,5,5, subsample=(2,2), activation='relu') )
model.add( Conv2D(36,5,5, subsample=(2,2), activation='relu') )
model.add( Conv2D(48,5,5, subsample=(2,2), activation='relu') )
model.add( Conv2D(64,3,3, activation='relu') )
model.add( Conv2D(64,3,3, activation='relu') )
model.add( Flatten() )
model.add( Dense(100) )
model.add( Dense(50) )
model.add( Dense(10) )
model.add( Dense(1) )
model.summary()
model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train,y_train,validation_split =0.2,shuffle=True,nb_epoch=5)
#

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=3, verbose=1)
model.save('model3.h5')
### print the keys contained in the history object
print(history_object.history.keys())

