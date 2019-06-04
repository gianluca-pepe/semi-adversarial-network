'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
This script contain the model of the FaceMatcher, is exactly the same defined during pretrain
exept that model is set as untrainable and weights are loaded
'''

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, \
    Activation, Concatenate
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K


class VGGFace():
    def __init__(self, input_shape=[224, 224, 1]):
        input_shape = [224, 224, 1]

        img_input = Input(shape=input_shape)

        img_conc = Concatenate()([img_input, img_input, img_input])

        x = ZeroPadding2D((1, 1), input_shape=input_shape)(img_conc)
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(128, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Convolution2D(4096, (7, 7), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Convolution2D(4096, (1, 1), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Convolution2D(2622, (1, 1))(x)
        y = Flatten()(x)
        y_last = Activation('softmax')(y)

        self.model = Model(inputs=img_input, outputs=y, name='FaceMatcher')

        self.model.load_weights(str(pathlib.Path('..', 'weights', 'facematcher_pretrain_weights.h5')))

        self.model.trainable = False

        self.model.compile(optimizer='Adam', loss=self.custom_loss)

    def custom_loss(self, target, output):
        return K.square(K.sqrt(K.sum(K.square(output - target), axis=1, keepdims=True) + K.epsilon()))
