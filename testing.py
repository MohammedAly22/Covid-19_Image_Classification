from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

classes = ["Covid", "Normal", "Viral Pneumonia"]
model = load_model('FinalCovidClassifier.h5')

def prepareImage(imagePath):
    image_size = 150
    img_array = cv.imread(imagePath)
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
    new_array = cv.resize(img_array, (image_size, image_size))
    plt.imshow(new_array)
    return new_array.reshape(-1, image_size, image_size, 3)

image = prepareImage("F:/normal.jpg")
prediction = model.predict(image)
index = np.argmax(prediction)
print(f"Prediction is {classes[index]}")
plt.show()
