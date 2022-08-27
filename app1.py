from flask import Flask,render_template,redirect,request,send_from_directory
import os
from PIL import Image
import numpy as np
import cv2
from torchvision import models
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from keras.models import load_model
from torch.utils import data
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow import keras
import keras.preprocessing
from keras.preprocessing.image import image_utils
from keras.preprocessing import image
from keras.utils import load_img


model_file = "Pneumonia Detection_model.h5"
model = load_model(model_file)
model.load_weights("Pneumonia Detection_model_weights.h5")
#model = torch.load(model_file)
model.summary()

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 



def makePredictions(path):
  '''
  Method to predict if the image uploaed is healthy or pneumonic
  '''
  img_d = load_img(path, target_size=(128, 128))
  tester_img = image_utils.img_to_array(img_d)/255
  tester_img1 = np.reshape(tester_img, (1, 128, 128, 3))


  predictions = model.predict(tester_img1).squeeze()
  print("************")
  print(predictions)
  a = int(np.rint(predictions))
  print(a)
  if a==1:
    a = "pneumonic"
  else:
    a = "healthy"
  return a

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        if 'img' not in request.files:
            return render_template('home.html',filename="unnamed.png",message="Please upload an file")
        f = request.files['img'] 
        filename = secure_filename(f.filename) 
        if f.filename=='':
            return render_template('home.html',filename="unnamed.png",message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html',filename="unnamed.png",message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files)==1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER']+'/'+file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return render_template('home.html',filename=f.filename,message=predictions,show=True)
    return render_template('home.html',filename='unnamed.png')

if __name__=="__main__":
    app.run(debug=True)