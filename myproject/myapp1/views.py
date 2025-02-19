from django.shortcuts import render
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import io
import numpy as np
# Create your views here.

model = load_model('myapp1/modelo_1.keras')

def index(request):
    # model = load_model('myproject/myapp1/modelo_1.keras')
    return render(request, 'myapp1/index.html')
    


def modelo(request):
    if 'reset' in request.POST:
        return render(request, 'myapp1/index.html')
    if request.method == 'POST' and 'image' in request.FILES:
        img_file = request.FILES['image']
        img = image.load_img(io.BytesIO(img_file.read()) , target_size=(200, 200))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0
        prediction = model.predict(img)
        prediction = np.argmax(prediction)
        labels = ['Neumonia', 'Normal']
        prediction_label = labels[prediction]
        return render(request, 'myapp1/index.html', {'prediction': prediction_label})
    return render(request, 'myapp1/index.html')