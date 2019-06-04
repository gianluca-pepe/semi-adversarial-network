'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
This script contain the model of the SAN (Semi-Adversarial-Network)
'''

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer,Conv2D, AveragePooling2D, UpSampling2D, Activation, Input, Concatenate
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.losses import mean_absolute_error
import pathlib
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.vis_utils import plot_model


# Custom loss function
'''
This class is used to compute te loss function as defined by the paper
The total loss in output is the sum of genderclassifier loss and the facematcher loss
'''
class TotalLoss(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super().__init__(**kwargs)

    def call(self, inputs):
        '''

        CLASSIFIER LOSS

        '''

        label = inputs[3]
        reversed_label = inputs[4]

        GP_sm = inputs[0]
        GP_op = inputs[1]

        same_gender_loss = K.categorical_crossentropy(label, GP_sm)
        opposite_gender_loss = K.categorical_crossentropy(reversed_label, GP_op)

        classifier_loss = same_gender_loss + opposite_gender_loss


        '''
        
        FACE MATCHER LOSS
        
        '''
        FM_img = inputs[5]
        FM_sm = inputs[2]

        face_matcher_loss = K.square(K.sqrt(K.sum(K.square(FM_sm - FM_img), axis=1, keepdims=True)+ K.epsilon()))

        '''
        
        TOTAL LOSS
        
        '''

        total_loss = classifier_loss + face_matcher_loss

        self.add_loss(total_loss, inputs=inputs)

        return total_loss # define the loss as output




class SemiAdversarial:
    def __init__(self, autoencoder, gender_classifier, face_matcher):

        gender_classifier.trainable = False
        face_matcher.trainable = False

        self.autoencoder = autoencoder
        self.models_dir = '../model'

        input_shape=[224, 224]


        # Inputs
        img = Input(shape=(224, 224, 1), name="Original_img")
        proto_sm = Input(shape=(224, 224, 3), name="SameGender_prototype")
        proto_op = Input(shape=(224, 224, 3), name="OppositeGender_prototype")

        label = Input(shape=(2,), name="Label")
        label_reversed = Input(shape=(2,), name="Label_reversed")
        vgg_ground = Input(shape=(2622,), name="vgg_ground")

        inputs = [img, proto_sm, proto_op, label, label_reversed, vgg_ground]

        #Here we obtain perturbed images from the autoencodet
        img_perturbed = self.autoencoder ([inputs[0],inputs[1],inputs[2]])

        #Here we obtain the prediction for gender according the same prototype
        GP_sm = gender_classifier (img_perturbed[0])

        #Gender prediction opposite gender
        GP_op = gender_classifier (img_perturbed[1])

        #Here we obtain the Face-Martcher feature vector
        FM = face_matcher (img_perturbed[0])

        #The total loss: in this case is the last layer of the SAN
        y_model = TotalLoss(name="Total_loss_custom")([GP_sm, GP_op, FM, inputs[3], inputs[4], inputs[5]])

        self.model = Model(inputs=inputs, outputs=y_model)


        print(self.model.summary())
        #plot_model(self.model, to_file='SemiAdversarialModelCustomLoss.png')


        #da creare loss custom
        self.model.compile(optimizer='adam', loss=None)



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
