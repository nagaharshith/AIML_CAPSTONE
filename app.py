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
from torch.utils import data
from PIL import Image

from werkzeug.utils import secure_filename

model_file = "PyTorch.h5"
#model = load_model(model_file)
model = torch.load(model_file)
model.eval()

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def makePredictions(path):
  '''
  Method to predict if the image uploaed is healthy or pneumonic
  '''
  #img = cv2.imread(path) # we open the image
  img=Image.open(path).convert('RGB')
  #img_d = cv2.resize(img,(224,224),3)
  img_preprocessed = preprocess(img)
  batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
  # we resize the image for the model
  ##rgbimg=None
  #We check if image is RGB or not
## if len(np.array(img_d).shape)<3:
##    rgbimg = Image.new("RGB", img_d.size)
##    rgbimg.paste(img_d)
## else:
##      rgbimg = img_d
##  rgbimg = np.array(rgbimg,dtype=np.float64)
##  rgbimg = rgbimg.reshape((1,224,224,3))

  #print(model.eval())
  predictions = model(batch_img_tensor)
  print("************")
 # print(predictions)
  print(predictions.detach().numpy()[0][0])
  a = int(np.rint(predictions.detach().numpy()[0][0]))
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