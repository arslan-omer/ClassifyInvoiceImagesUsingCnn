from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image 
import numpy as np
import json
import glob, os
import cv2

img_width, img_height = 224, 224

with open('D:\\Invoice_Data_Test\\saved_cnn_model.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('D:\\Invoice_Data_Test\\saved_cnn_model.h5')
result_text = ''

result_text = result_text + '==========================================================\n'
result_text = result_text + 'Test dizinine eklenen fatura kategoriler tahmin ediliyor...\n'
result_text = result_text + 'Tahmin edilecek kategoriler :\n'
result_text = result_text + '0 --> Kart Dolum Fişi\n'
result_text = result_text + '1 --> Otopark Fişi\n'
result_text = result_text + '2 --> Taksi Fişi\n'
result_text = result_text + '3 --> Yemek Fişi\n'
result_text = result_text + '==========================================================\n'

prediction_path  = "D:\\Invoice_Data_Test\\Prediction"
os.chdir(prediction_path)
for filePath in glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg"):
    img = image.load_img(filePath, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    result_text = result_text + 'Ağa Verilen Görüntü --> ' + filePath
    result_text = result_text + ' Bulunan sınıf: ' +  str(classes[0])
    if(classes[0]==0):
        result_text = result_text + '---> Bulunan sınıf ismi  --->  Kart Dolum Fişi\n'
    if(classes[0]==1):
        result_text = result_text + '---> Bulunan sınıf ismi  --->  Otopark Fişi\n'
    if(classes[0]==2):
        result_text = result_text + '---> Bulunan sınıf ismi  --->  Taksi Fişi\n'
    if(classes[0]==3):
        result_text = result_text + '---> Bulunan sınıf ismi  --->  Yemek Fişi\n'

file_obj = open('D:\\Invoice_Data_Test\\Sonuclar.txt','w')
file_obj.write(result_text)

print('İşlem tamamlandı...')