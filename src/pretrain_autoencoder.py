'''
This script has been created by Bruno Marino and Gianluca Pepe
'''

from datetime import datetime
from dataset_loader import DatasetLoader
from modules.autoencoder import AutoEncoder
from os import path


def save_weights(model, name):
    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = str(filename) + ".h5"
    filename = name + '-' + filename
    filename = path.join('..', 'weights', filename)
    model.save_weights(filename)


ds = DatasetLoader()
itr = ds.iterator_AE()
itr_val = ds.iterator_AE_VAL()

NN = AutoEncoder()

try:
    NN.model.fit_generator(generator=itr, validation_data=itr_val,
                           validation_steps=426,
                           steps_per_epoch=2355,
                           epochs=10, verbose=1, workers=0)
except KeyboardInterrupt:
    # salvo il modello aggiungendo al nome quando e' stato creato
    save_weights(NN.model, 'autoencoder')
else:
    save_weights(NN.model, 'autoencoder')
