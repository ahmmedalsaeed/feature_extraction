from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet
from keras.preprocessing.image import load_img ,img_to_array
from tensorflow.keras.applications.vgg19 import VGG19
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
            self.Base_model =VGG19(weights='imagenet')

        elif modelName == ("EfficientNet"):
            self.Base_model=EfficientNet.from_pretrained("efficientnet-b0")
   
     def get_name(self):
       return self.Base_model
     
     def extract(self,image):
       img = load_img(image, target_size=(224, 224))
       x = img_to_array(img)
       x = np.expand_dims(x, axis=0)
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

     
img="/content/cat.jpg"
efficientnet=Feature_extraction('VGG19')
features=efficientnet.extract(img)
print(features)