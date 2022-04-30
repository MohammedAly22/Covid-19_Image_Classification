############# IMPORT PACKAGES ##############
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

############# DEFINE IMAGES PATHS ##############
train_image_path = "Covid19-dataset/train"
test_image_path = "Covid19-dataset/test"

############# IMAGE EXPLORATION ##############
img = plt.imread(os.path.join(train_image_path, "Covid/01.jpeg"))
plt.imshow(img)
height, width, dim = img.shape
plt.title('Covid')
print("size of image (h x w x d)", height, width, dim)
plt.show()

img = plt.imread(os.path.join(train_image_path, "Viral Pneumonia/01.jpeg"))
plt.imshow(img)
height, width, dim = img.shape
plt.title('Viral Pneumonia')
print("size of image (h x w x d)", height, width, dim)
plt.show()

img = plt.imread(os.path.join(train_image_path, "Normal/01.jpeg"))
plt.imshow(img)
height, width, dim = img.shape
plt.title('Normal')
print("size of image (h x w x d)", height, width, dim)
plt.show()

############# IMAGE AUGMENTATION ##############
train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest')

train_dataset = train.flow_from_directory(
    train_image_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

test = ImageDataGenerator(
    rescale=1. / 255)

test_dataset = test.flow_from_directory(
    test_image_path,
    target_size=(150, 150),
    batch_size=32,
    shuffle=False)

print(train_dataset.class_indices)

############# MODEL BUILDING ##############
model = Sequential()
model.add(Conv2D(128, kernel_size=6, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

############# MODEL COMPILING ##############
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

############# MODEL TRAINING(FITTING) ##############
steps_per_epoch = np.math.ceil(train_dataset.samples / train_dataset.batch_size)
epochs = 100
history = model.fit_generator(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_dataset
)

############# PLOTTING LOSS AND ACCURACY ##############
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Training Vs. Validation Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Training Vs. Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

############# GENERATING PREDICTIONS ##############
test_steps_per_epoch = np.math.ceil(test_dataset.samples / test_dataset.batch_size)
loss_accuracy = model.evaluate(test_dataset)
print(f"Loss: {loss_accuracy[0]}")
print(f"Accuracy: {loss_accuracy[1]}")
predictions = model.predict(test_dataset, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)

############# CLASSIFICATION REPORT ##############
true_classes = test_dataset.classes
class_labels = list(test_dataset.class_indices.keys())
print("Classification Report: ")
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

############# CONFUSION MATRIX ##############
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(7, 7))
sns.heatmap(conf_matrix, annot=True)
plt.show()

############# MODEL SAVING ##############
model.save('FinalCovidClassifier.h5')
