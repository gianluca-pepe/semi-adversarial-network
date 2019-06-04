'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
Here we perform the the train of the SAN
we save both weigths of the entire newtork and weights if the autoencoder

Only weights of the autoencoder are useful given that is the only trainable component of the SAN
and we use these weingts during the evaluation phase.

Weights of the SAN contain all weights of all the SAN components: the pre-loaded weights of the untrainable components and the weights of the autoencoder.
'''

from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from os import path
from modules.autoencoder import AutoEncoder
from modules.genderclassifier import GenderClassifier
from modules.facematcher import VGGFace
from modules.san import SemiAdversarial
from dataset_loader import DatasetLoader


def save_weights(model,name):
    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = str(filename) + ".h5"
    filename = name + '-' + filename
    filename = path.join('..', 'weights', filename)
    model.save_weights(filename)


AE = AutoEncoder(mode='further_train')
GC = GenderClassifier()
FM = VGGFace()

ds = DatasetLoader()
itr = ds.iterator_further(FM.model)
itr_val = ds.iterator_further_VAL(FM.model)

# Semi-Adversarial-Netowrk
SAN = SemiAdversarial(AE.model, GC.model, FM.model)

try:

    train = SAN.model.fit_generator(generator=itr, validation_data=itr_val,
                                    validation_steps=63,  # 852
                                    steps_per_epoch=313,  # 4410
                                    epochs=10, verbose=1, workers=0, )

    #print(train.history.keys())

    loss_history = []
    for arr in train.history['loss']:
        loss_history.append(np.average(arr))

    val_loss_history = []
    for arr in train.history['val_loss']:
        val_loss_history.append(np.average(arr))

    # Plotting losses
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    plt.savefig('loss_plot.jpg')

except KeyboardInterrupt:
    # Save the autoencoder weights
    save_weights(SAN.autoencoder, 'autoencoder')

    # Save the semiadversarial weights
    save_weights(SAN.model, 'SemiAdversarial')


else:
    # Save the autoencoder weights
    save_weights(SAN.autoencoder, 'autoencoder')

    # Save the semiadversarial weights
    save_weights(SAN.model, 'SemiAdversarial')
