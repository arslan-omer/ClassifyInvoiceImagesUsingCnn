import cv2
from matplotlib import pyplot
import glob, os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import numpy as np
processed_image_order = 0

path_list = [
"D:\\Invoice_Data_Test\\Prediction"
]

image_width = 510
image_height = 710

for main_directory in path_list:
    os.chdir(main_directory)
    for filePath in glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg"):
        #önce görüntüyü okuyoruz
        img = cv2.imread(filePath)
        ww = 2120
        hh = 2120
        color = (0,0,0)
        result = np.full((hh,ww,3), color, dtype=np.uint8)
        # compute center offset
        xx = (ww - image_width) // 2
        yy = (hh - image_height) // 2
        # copy img image into center of result image
        result[yy:yy+image_height, xx:xx+image_width] = img
        img = result
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(
	                                  brightness_range=[0.2,1.0],
	                                  rotation_range=90
                                    )
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        processed_image_order = 1
        for i in range(9):
            batch = it.next()
            file_name = os.path.splitext(filePath)[0]
            extension = os.path.splitext(filePath)[1]
            new_file_name = file_name + '_augmented_' + str(processed_image_order) + extension
            image = batch[0].astype('uint8')
            cv2.imwrite(new_file_name, image)
            processed_image_order = processed_image_order + 1

        (h, w) = img.shape[:2]
        
        # calculate the center of the image
        center = (w / 2, h / 2)
        angle90 = 90
        angle180 = 180
        angle270 = 270
        scale = 1.0

        # Perform the counter clockwise rotation holding at the center

        # 90 degrees
        M = cv2.getRotationMatrix2D((w/2,w/2), angle90, scale)
        rotated90 = cv2.warpAffine(img, M, (h, w))

        # 180 degrees
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(img, M, (w, h))
       
        # 270 degrees
        M = cv2.getRotationMatrix2D((h/2,h/2), angle270, scale)
        rotated270 = cv2.warpAffine(img, M, (h, w))
        
        processed_image_order = processed_image_order + 1
        new_file_name = file_name + '_augmented_rotated_' + str(processed_image_order) + extension
        cv2.imwrite(new_file_name,img)
        processed_image_order = processed_image_order + 1
        new_file_name = file_name + '_augmented_rotated_' + str(processed_image_order) + extension
        cv2.imwrite(new_file_name,rotated90)
        processed_image_order = processed_image_order + 1
        new_file_name = file_name + '_augmented_rotated_' + str(processed_image_order) + extension
        cv2.imwrite(new_file_name,rotated180)
        processed_image_order = processed_image_order + 1
        new_file_name = file_name + '_augmented_rotated_' + str(processed_image_order) + extension
        cv2.imwrite(new_file_name,rotated270)
        processed_image_order = processed_image_order + 1
        print('işlem devam ediyor :)')