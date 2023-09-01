from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_model(Image_Width, Image_Height, Image_Channels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Image_Width, Image_Height, Image_Channels)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    return model, callbacks


def training_validation(batch_size, Image_Size):
    train_datagen = ImageDataGenerator(rotation_range=15,
                                       rescale=1. / 255,
                                       shear_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1
                                       )
    train_generator = train_datagen.flow_from_directory("./train/",
                                                        target_size=Image_Size,
                                                        class_mode='categorical',
                                                        batch_size=batch_size)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        "./validation/",
        target_size=Image_Size,
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator, validation_generator


def model_fit(model, train_generator, validation_generator, batch_size, callbacks, epochs, total_validate, total_train):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        batch_size=batch_size
    )

    model.save("models-new/change-test-20epoch-128-batch32-noBatch-512.h5")

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('graphs-new/acc-20e-128-noBatch-512.png')
    plt.show()

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('graphs-new/loss-20e-128-noBatch-512.png')
    plt.show()
