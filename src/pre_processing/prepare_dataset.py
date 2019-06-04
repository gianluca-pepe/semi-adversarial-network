'''
This script has been created by Bruno Marino and Gianluca Pepe
'''

import os
import random
import shutil
from prototype_generation import create_prototypes


# Here we create the validation set starting from the training set.
# We decide the size of validation as 15% of training set
def create_validation(src, dst):
    print('--- Validation set pre-processing ---')
    labels = ['male', 'female']

    for gender in labels:

        path = os.path.join(src, gender)
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

        random.shuffle(files)

        # move the 15% of each directory files to validation respective directory
        num_15 = int(len(files) * 0.15)

        for i in range(num_15):
            shutil.move(os.path.join(path, files[i]), os.path.join(dst, gender, files[i]))

        print('Validation set: ' + gender + ' images processed')


# This function organizes the dataset, creating train and test folders
# wich contains the training-set and the test.set respectively.
def split_images(src, dst):
    types = ['train', 'test']
    # preparing the dataset
    for type in types:
        print('--- ' + type + ' set pre-processing ---')
        count = 0
        src = src + type
        dst = os.path.join(dst, type)

        with open("list_attr_celeba_ground.txt") as f:
            for line in f:
                row = line.split(' ')

                name = row[0]
                gender = row[21]

                # gender=-1 => female
                # gender=1 => male
                if gender == "1":
                    dst_img_path = os.path.join(dst, "male", name)

                else:
                    dst_img_path = os.path.join(dst, "female", name)

                src_img_path = os.path.join(src, name)

                if (count % 10000) == 0:
                    print(count)

                if os.path.isfile(src_img_path):
                    shutil.copyfile(src_img_path, dst_img_path)
                    count = count + 1

        print("The " + type + " set contain", str(count), " samples")


REPO_ROOT = os.path.join('..', '..')
SRC = os.path.join(REPO_ROOT, 'images-dpmcrop-')
DST = os.path.join(REPO_ROOT, 'dataset')
DST_TRAIN = os.path.join(DST, 'train')
DST_VALID = os.path.join(DST, 'validation')
DST_PROTO = os.path.join(DST, 'prototype')

try:
    os.mkdir(DST)
    os.makedirs(os.path.join(DST, 'train', 'male'))
    os.makedirs(os.path.join(DST, 'train', 'female'))
    os.makedirs(os.path.join(DST, 'test', 'male'))
    os.makedirs(os.path.join(DST, 'test', 'female'))

    os.makedirs(os.path.join(DST_VALID, 'male'))
    os.makedirs(os.path.join(DST_VALID, 'female'))
    os.makedirs(os.path.join(DST_PROTO, 'male'))
    os.makedirs(os.path.join(DST_PROTO, 'female'))
except OSError:
    print('something went wrong or dataset folder already exists')
else:
    print('dataset folder and subfolders created successfully')

# split images in male and female folder for both train and test set
split_images(SRC, DST)
# Create validation set from training set
create_validation(DST_TRAIN, DST_VALID)
# generate the three prototype images
create_prototypes(DST_TRAIN, DST_PROTO)
