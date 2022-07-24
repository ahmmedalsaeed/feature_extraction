from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet
import keras
from tensorflow.keras.utils import img_to_array,load_img
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
import argparse

class Feature_extraction:
     

     def __init__(self, modelName):
        self.modelName = modelName
        if modelName == ("VGG19"):
            #self.Base_model =VGG19(weights='feature_extraction/vgg19_weights.h5')
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
           features = features.detach().numpy()
           #df = pd.DataFrame(features)
           #df.to_csv ('export_dataframe'+self.modelName+'.csv', index = None, header=True)
       return features
     def get_sammary(self):

        return self.Base_model.summary()
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, required=True)
parser.add_argument('--model', type=str, required=True)

args = parser.parse_args()
     
img="feature_extraction\cat.jpg"
VGG=Feature_extraction(args.model)
features=VGG.extract(args.img_path)
print(features)