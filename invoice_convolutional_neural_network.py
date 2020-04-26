from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.preprocessing import image 
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import json
import os

img_width, img_height = 400, 600
  
train_data_dir = "D:\\Invoice_Data_Test\\Train"
validation_data_dir = "D:\\Invoice_Data_Test\\Test"
nb_train_samples = 1288
nb_validation_samples = 560
epochs = 20
batch_size = 16
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
  
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(4)) 
model.add(Activation('sigmoid')) 
  
model.compile(loss ='binary_crossentropy', 
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 
  
train_datagen = ImageDataGenerator( 
                                     rescale = 1. / 255
                                  ) 
  
test_datagen = ImageDataGenerator( 
                                    rescale = 1. / 255
                                  ) 
 
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='categorical') 
  
validation_generator = test_datagen.flow_from_directory( 
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