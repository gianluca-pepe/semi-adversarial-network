'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
This script contain the model of the GenderClassifier
'''

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import pathlib


class GenderClassifier():
    def __init__(self, mode='pretrain', input_shape=[224, 224, 1]):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=5, input_shape=input_shape))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(64, kernel_size=3))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(64, kernel_size=3))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(128, kernel_size=3))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(128, kernel_size=3))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(256, kernel_size=3))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Dense(256))

        self.model.add(Dropout(0.5))

        self.model.add(LeakyReLU())

        self.model.add(Dense(2))

        self.model.add(Activation('sigmoid', name='pred_output'))

        self.model.add(Flatten())

        if mode != 'pretrain':
            # Load weights for this model
            self.model.load_weights(str(pathlib.Path('..', 'weights', 'genderclassifier_pretrain_weights.h5')))
            self.model.trainable = False


        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        print(self.model.summary())

    def savemodel(self, name):
        if name.endswith('.h5'):
            filename = name
        else:
            filename = pathlib.Path(self.models_dir, name + '.h5')
        self.model.save(filename)
        print("\nModel saved successfully on file %s\n" % filename)

    def loadmodel(self, name):
        if name.endswith('.h5'):
            filename = name
        else:
            filename = pathlib.Path(self.models_dir, name + '.h5')
        try:
            self.model = Model.load_model(filename)
            print("\nModel loaded successfully from file %s\n" % filename)
        except OSError:
            print("\nModel file %s not found!!!\n" % filename)
            self.model = None
