from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet
import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.EfficientNetB0 import EfficientNetB0
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Model
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
class Feature_extraction:

     def __init__(self, modelName):
        self.modelName = modelName
        if modelName == ("VGG19"):
            self.Base_model =VGG19(weights='feature_extraction/vgg19_weights.h5')
            #self.Base_model =Base_model(weights='imagenet')
        elif modelName == ("EfficientNet"):
            self.Base_model=EfficientNet.from_pretrained("efficientnet-b0")
   
     def get_name(self):
       return self.Base_model
     
     def extract(self,image):
       img=cv2.imread(image)
       img = cv2.resize(img,(224,224))
       img = img.reshape(1,224,224,3)
       x = np.asarray( img, dtype="int32" )
       x = preprocess_input(x)
       if self.modelName == 'VGG19':
            
            model = Model(inputs=self.Base_model.inputs, outputs=self.Base_model.layers[-2].output)
            print(model.summary())
            features = model.predict(x)
            df = pd.DataFrame(features)
            df.to_csv ('export_dataframe'+self.modelName+'.csv', index = None, header=True) 

       if self.modelName == 'EfficientNet':
           img = Image.open(image).convert('RGB')

           resize = transforms.Resize([224, 224])
           img = resize(img)
           to_tensor = transforms.ToTensor()
           tensor = to_tensor(img)
           tensor = tensor.unsqueeze(0)
           features = self.Base_model.extract_features(tensor)
       
       return features
     def get_sammary(self):

        return self.Base_model.summary()

     
img="feature_extraction\cat.jpg"
VGG=Feature_extraction('VGG19')
features=VGG.extract(img)
print(features)