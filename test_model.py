import os
import keras
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array


model_path = '/home/agrivision/detection/model/predictor.h5'

model = keras.models.load_model(model_path)

print("@@ Model Loaded")

hu = '/home/agrivision/detection/Dataset/testing/hu.png'
ho1 = '/home/agrivision/detection/Dataset/testing/ho1.jpeg'
ho2 = '/home/agrivision/detection/Dataset/testing/ho2.jpeg'

def predict (horse_human):
    test_image = image.load_img(horse_human, target_size = (150, 150))
    print("@@ Image loaded for prediction")
    test_image = image.img_to_array(test_image)/255 #convrt into np_array
    test_image = np.expand_dims(test_image, axis=0) #change dimention 3D to 4D
    
    
    result = model.predict(test_image).round(3)
    print("@@ result ", result)
    
    pred = np.argmax(result) #get the index of maximum value
    print(result, "====>", pred)
    
    if pred == 0:
        print("it's a Horse")
    else:
        print("it's a Human")
        
        
#for horse_human in [hu, ho1, ho2]:
 #   predict(model, horse_human)
    
    
    
    
#////////////////////////////////////////////////////////////
# create flGETask instance 

app = Flask(__name__)

#render index.html page

@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/prediction", methods = ['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print("@@input posted", filename) 
        
        file_path = os.path.join('/user_uploaded', filename)
        file.save(file_path)
        
        print("@@prediction class....")
        pred = predict(horse_human=filepath)
        return render_template('prediction.html',  pred_output = pred, user_image = file_path)
if __name__ == "__main__":
    app.run(threaded=False)