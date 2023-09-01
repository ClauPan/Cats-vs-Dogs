import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def create_test(batch_size, Image_Size):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory("./test_set/",
                                                      target_size=Image_Size,
                                                      class_mode='categorical',
                                                      batch_size=batch_size)

    return test_generator


def draw_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['cat', 'dog'])
    plt.yticks([0, 1], ['cat', 'dog'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()


def show_accuracy(model, generator):
    score = model.evaluate(generator, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def test_img(model, Image_Size):
    print("To test an image you can either add it to the folder test, or chose an existing one from the same folder. \n")
    print("To evaluate an image, write its name. To stop write 'end'.\n")
    string = input()
    while string != "end":
        im = cv2.imread("./test/" + string + ".jpg")
        im = cv2.resize(im, Image_Size)
        im = np.expand_dims(im, axis=0)
        im = np.array(im)
        im = im / 255
        pred = model.predict(im)[0][1]
        print(pred)
        pred = np.where(pred > 0.5, 1, 0)
        if pred == 0:
            print("Your image is a cat")
        else:
            print("Your image is a dog")
        string = input()
