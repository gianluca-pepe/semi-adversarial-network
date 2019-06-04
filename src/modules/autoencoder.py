'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
This script contain the model of the AutoEncoder
'''

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, UpSampling2D, Activation, Input, Concatenate
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import pathlib
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.vis_utils import plot_model


class AutoEncoder:
    def __init__(self, mode='pretrain', input_shape=[224, 224]):

        self.models_dir = pathlib.Path('..', '..', 'model')

        # Inputs
        img = Input(shape=(224, 224, 1))
        proto_sm = Input(shape=(224, 224, 3))
        proto_op = Input(shape=(224, 224, 3))

        if mode == 'pretrain':
            inputs = [img, proto_sm]
        else:
            inputs = [img, proto_sm, proto_op]

        x = Concatenate()([img, proto_sm])

        # Encoder
        x = Conv2D(8, kernel_size=3, padding='SAME', input_shape=input_shape)(x)
        x = LeakyReLU()(x)
        x = AveragePooling2D(pool_size=2, strides=2)(x)

        x = Conv2D(12, kernel_size=3, padding='SAME')(x)
        x = LeakyReLU()(x)
        x = AveragePooling2D(pool_size=2, strides=2)(x)

        # Decoder
        x = Conv2D(256, kernel_size=3, padding='SAME')(x)
        x = LeakyReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        x = Conv2D(128, kernel_size=3, padding='SAME')(x)
        x = LeakyReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        # final shared layers
        output = Conv2D(1, kernel_size=1, padding='VALID')
        activation = Activation('sigmoid')

        # Protocombiner
        y_sm = Concatenate()([x, proto_sm])
        y_sm = output(y_sm)
        y_sm = activation(y_sm)

        if mode != 'pretrain':
            y_op = Concatenate()([x, proto_op])
            y_op = output(y_op)
            y_op = activation(y_op)

            self.model = Model(inputs=inputs, outputs=[y_sm, y_op], name="AutoEncoder")
            self.model.load_weights(str(pathlib.Path('..', 'weights', 'autoencoder_pretrain_weights.h5')))
        else:
            self.model = Model(inputs=inputs, outputs=y_sm, name="AutoEncoder")

        self.model.compile(optimizer='adadelta', loss=self.pixelwise_crossentropy)

        # Plot and Summarize
        # plot_model(self.model, 'autoencoder.png')
        print(self.model.summary())

    # Custom Loss function
    def pixelwise_crossentropy(self, target, output):
        output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
        return - tf.reduce_sum(target * tf.log(output))

    def savemodel(self, name):
        filename = pathlib.Path(self.models_dir, name + '.h5')

        try:
            self.model.save(filename)

            print("\nModel saved successfully on file %s\n" % filename)
        except OSError:
            print("\nUnable to save the model on file %s\n Please check the path and/or permissions" % filename)

    def loadmodel(self, name):
        filename = pathlib.Path(self.models_dir, name + '.h5')

        try:
            self.model = Model.load_model(filename)

            print("\nModel loaded successclass_weightfully from file %s\n" % filename)
        except OSError:
            print("\nModel file %s not found!!!\n" % filename)
            self.model = None
