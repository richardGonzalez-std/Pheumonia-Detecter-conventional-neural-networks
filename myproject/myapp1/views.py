from django.shortcuts import render
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# Create your views here.

model = load_model('D:/Code_/myproject/myapp1/modelo_1.keras')

def index(request):
    # model = load_model('myproject/myapp1/modelo_1.keras')
    return render(request, 'myapp1/index.html')
    


def modelo(request, model=model):
    if request.method == 'POST':
        img = request.FILES['imagen']
        img = image.load_img(img, target_size=(200, 200))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255
        prediction = model.predict(img)
        prediction = np.argmax(prediction)
        if prediction == 0:
            prediction = 'Neumonia'
        else:
            prediction = 'Normal'
        return render(request, 'myapp1/index.html', {'prediction': prediction})
    return render(request, 'myapp1/index.html')