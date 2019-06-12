'''
This script has been created by Bruno Marino and Gianluca Pepe
'''

from datetime import datetime
from tensorflow.python import keras
from os import path
from utils.dataset_loader import DatasetLoader
from modules.genderclassifier import GenderClassifier
from utils.util import save_weights

ds = DatasetLoader()
itr = ds.iterator_GC()
itr_val = ds.iterator_GC_VAL()

callback = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='auto',
                                         baseline=98, restore_best_weights=False)

NN = GenderClassifier()

try:

    NN.model.fit_generator(generator=itr, validation_data=itr_val,
                           validation_steps=416,  # 416
                           steps_per_epoch=2355,  # 2355
                           epochs=10, verbose=1, workers=0, callbacks=[callback])  # 10 epoch
except KeyboardInterrupt:
    save_weights(NN.model, 'genderclassifier')
else:
    save_weights(NN.model, 'genderclassifier')
