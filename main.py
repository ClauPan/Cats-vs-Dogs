import numpy as np
import os
import cv2


import model as nn
import keras.models
import test as t

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3
batch_size = 32
epochs = 20


model, callbacks = nn.create_model(Image_Width, Image_Height, Image_Channels)
train_generator, validation_generator = nn.training_validation( batch_size, Image_Size)
#nn.model_fit(model, train_generator, validation_generator, batch_size, callbacks, epochs, total_validate, total_train)

model_fit = keras.models.load_model("models-new/change-test-20epoch-128-batch32-noBatch-512.h5")

test_generator = t.create_test(batch_size, Image_Size)

y_true = test_generator.classes

x_test = []

for folder in os.listdir("test_set"):
    sub_path = "test_set" + "/" + folder
    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, Image_Size)
        x_test.append(img_arr)

test_x = np.array(x_test)
test_x = test_x / 255.0
            

y_pred = model_fit.predict(test_x)

y_pred = np.argmax(y_pred, axis=1)

t.draw_matrix(y_true, y_pred)

t.show_accuracy(model_fit, test_generator)

t.test_img(model_fit, Image_Size)
