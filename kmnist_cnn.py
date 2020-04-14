from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import backend as K
import keras
import numpy as np

num_classes = 10
img_rows, img_cols = 28, 28


def load(f):
    return np.load(f)['arr_0']


X_train = load('dataset/kmnist-train-imgs.npz')
X_test = load('dataset/kmnist-test-imgs.npz')
y_train = load('dataset/kmnist-train-labels.npz')
y_test = load('dataset/kmnist-test-labels.npz')

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32') / 255
    X_test = X_train.astype('float32') / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)


def main():
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, batch_size=32, nb_epoch=100)

    # モデルの保存
    model.sava('./kmnist_cnn.h5')


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])


if __name__ == "__main__":
    main()
