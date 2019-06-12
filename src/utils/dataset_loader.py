'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
This script is used to build the iterators used by Keras for training singolar models (Pre-Train) and the final 
Semi-Adversarial model (Further-Train)
'''

import tensorflow as tf
import pathlib, random, os
from tensorflow.python.keras import backend as K
# provide one hot encoding of the label
from tensorflow.python.keras.utils import to_categorical
from utils.util import RepoPaths

paths = RepoPaths()
# Path of the dataset
DATASET_PATH = paths.ds_celeb

LABELS = dict({'female': 0, 'male': 1})

# Prototypes path
FEMALE_PATH = os.path.join(paths.proto, 'female.jpg')
MALE_PATH = os.path.join(paths.proto, 'male.jpg')
NEUTRAL_PATH = os.path.join(paths.proto, 'neutral.jpg')

SAMEPROTOTYPE = dict({'female': FEMALE_PATH, 'male': MALE_PATH})
OPPOSITEPROTOTYPE = dict({'female': MALE_PATH, 'male': FEMALE_PATH})


# Load and normalize images in tensor
def _parse_function(f1, f2, f3, f4, label):
    # original in balck/white

    img1_string = tf.read_file(f1)

    img1_decoded = tf.image.decode_jpeg(img1_string, channels=1)

    img1 = tf.cast(img1_decoded, tf.float32) / tf.constant(255.0)

    # same gender RGB

    img2_string = tf.read_file(f2)

    img2_decoded = tf.image.decode_jpeg(img2_string, channels=3)

    img2 = tf.cast(img2_decoded, tf.float32) / tf.constant(255.0)

    # opposite gender RGB

    img3_string = tf.read_file(f3)

    img3_decoded = tf.image.decode_jpeg(img3_string, channels=3)

    img3 = tf.cast(img3_decoded, tf.float32) / tf.constant(255.0)

    # neutral RGB

    img4_string = tf.read_file(f4)

    img4_decoded = tf.image.decode_jpeg(img4_string, channels=3)

    img4 = tf.cast(img4_decoded, tf.float32) / tf.constant(255.0)

    return img1, img2, img3, img4, label


# --------------------ITERATORS-------------------------------------------------

'''
batch[0] = original image
batch[1] = same gender prototype
batch[2] = opposite gender prototype
batch[3] = neutral gender prototype
batch[4] = binary label (male or female)
'''


# iterator autoencoder (pretrain)

def make_iterator_AE(dataset):
    iterator = dataset.make_one_shot_iterator()

    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            batch = sess.run(next_val)

            yield [batch[0], batch[1]], batch[0]


# Iterator gender-classifier (pretrain)

def make_iterator_GC(dataset):
    iterator = dataset.make_one_shot_iterator()

    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            batch = sess.run(next_val)

            yield batch[0], batch[4]


'''
ITERATOR FOR FURTHER TRAIN
'''


def make_iterator_further(dataset, vgg=None):
    iterator = dataset.make_one_shot_iterator()

    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:

        while True:

            batch = sess.run(next_val)

            reversed_label = batch[4][::-1]

            if vgg is None:

                dummy_array = np.zeros(2622, dtype=np.float32)

                yield [batch[0], batch[1], batch[2], batch[4], reversed_label, dummy_array], None

            else:

                FM_img = vgg.predict(batch[0], batch_size=32, steps=1)

                yield [batch[0], batch[1], batch[2], batch[4], reversed_label, FM_img], None


# -------------------------------------------------------------------


'''
        CLASS DATASET LOADER
'''
'''
This class is used to obtain the iterators
'''


class DatasetLoader:

    def __init__(self, modality="train"):

        self.batch_size = 32
        self.ds_val_root = paths.celeba['valid']

        if modality == "train" or modality == "test":
            self.ds_root = paths.celeba[modality]
        else:
            raise TypeError("Dataset modality can be only 'train' or 'test', other values are not accepted.")

        all_image_paths = []
        all_image_labels = []
        all_image_same = []
        all_image_neutral = []
        all_image_opposite = []

        # build an array of paths (relative paths)

        for label in pathlib.Path(self.ds_root).iterdir():

            for image_path in pathlib.Path(self.ds_root, label.name).iterdir():
                all_image_paths.append(str(image_path))

        random.shuffle(all_image_paths)

        # Consider training-set of 10000 element
        all_image_paths = all_image_paths[:10000]

        # load labels and image path of same, opposite and neutral gender

        for path in all_image_paths:
            dir_name = pathlib.Path(path).parent.name

            all_image_labels.append(LABELS[dir_name])

            all_image_same.append(SAMEPROTOTYPE[dir_name])

            all_image_opposite.append(OPPOSITEPROTOTYPE[dir_name])

            all_image_neutral.append(NEUTRAL_PATH)

        all_image_labels = to_categorical(all_image_labels)

        self.dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_same, all_image_opposite,

                                                           all_image_neutral, all_image_labels))

        # Load the images to tensors
        self.dataset = self.dataset.map(_parse_function)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(32)

        # --------- VALIDATION ----------------

        all_image_paths = []
        all_image_labels = []
        all_image_same = []
        all_image_neutral = []
        all_image_opposite = []

        # build an array of paths (relative paths)
        for label in pathlib.Path(self.ds_val_root).iterdir():
            for image_path in pathlib.Path(self.ds_val_root, label.name).iterdir():
                all_image_paths.append(str(image_path))

        random.shuffle(all_image_paths)

        # Consider validation-set of 2000 element
        all_image_paths = all_image_paths[:2000]

        # load labels and image path of same, opposite and neutral gender

        for path in all_image_paths:
            dir_name = pathlib.Path(path).parent.name

            all_image_labels.append(LABELS[dir_name])

            all_image_same.append(SAMEPROTOTYPE[dir_name])

            all_image_opposite.append(OPPOSITEPROTOTYPE[dir_name])

            all_image_neutral.append(NEUTRAL_PATH)

        all_image_labels = to_categorical(all_image_labels)

        self.val_data = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_same, all_image_opposite,

                                                            all_image_neutral, all_image_labels))

        # Load the images to tensors
        self.val_data = self.val_data.map(_parse_function)

        self.val_data = self.val_data.repeat()

        self.val_data = self.val_data.batch(32)

        # -----------------------------------------

    # AutoEncoder Iterator
    def iterator_AE(self):

        itr = make_iterator_AE(self.dataset)

        return itr

    # Gender-Classifier Iterator
    def iterator_GC(self):

        itr = make_iterator_GC(self.dataset)

        return itr

    # Semi-Adversarial model Iterator (Further-Training)
    def iterator_further(self, vgg=None):

        itr = make_iterator_further(self.dataset, vgg)

        return itr

    # Semi-Adversarial model Iterator (Further-Training) (Validation)
    def iterator_further_VAL(self, vgg=None):

        itr = make_iterator_further(self.val_data, vgg)

        return itr

    # Gender-Classifier Iterator	(Validation)
    def iterator_GC_VAL(self):

        itr = make_iterator_GC(self.val_data)

        return itr

    # AutoEncoder Iterator (Validation)
    def iterator_AE_VAL(self):

        itr = make_iterator_AE(self.val_data)

        return itr
