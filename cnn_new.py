from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D,MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.preprocessing import image 
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import json
import os

img_width, img_height = 224, 224
  
train_data_dir = "D:\\Invoice_Data_Test\\Train"
validation_data_dir = "D:\\Invoice_Data_Test\\Test"
nb_train_samples = 1288
nb_validation_samples = 560
epochs = 5
batch_size = 16

# Layer Values
num_filters = 32            # No. of conv filters
max_pool_size = (2,2)       # shape of max_pool
conv_kernel_size = (3, 3)    # conv kernel shape
imag_shape = (224, 224, 3)
num_classes = 3
drop_prob = 0.5

# Define model type
model = Sequential()
# 1st Layer
model.add(Conv2D(num_filters, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))
# 2nd Convolution Layer
model.add(Conv2D(num_filters*2, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))
# 3nd Convolution Layer
model.add(Conv2D(num_filters*4, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))
#Fully Connected Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))   #Fully connected layer
# Dropout some neurons to reduce overfitting
model.add(Dropout(drop_prob))
#Readout Layer
model.add(Dense(num_classes, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
datagen = image.ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=False)
datagen_v = image.ImageDataGenerator(rescale=1./255)  


train_generator = datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='categorical') 
  
validation_generator = datagen_v.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='categorical') 

model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size  
    ) 


model_json = model.to_json()
with open('D:\\Invoice_Data_Test\\saved_cnn_model.json', "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights('D:\\Invoice_Data_Test\\saved_cnn_model.h5') 
