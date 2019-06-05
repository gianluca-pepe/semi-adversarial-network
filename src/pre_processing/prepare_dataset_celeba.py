'''
This script has been created by Bruno Marino and Gianluca Pepe
'''

import os
import random
import shutil
from prototype_generation import create_prototypes
from repo_paths import RepoPaths


# Here we create the validation set starting from the training set.
# We decide the size of validation as 15% of training set
def create_validation(paths_celeba):
    print('--- Validation set pre-processing ---')
    labels = ['male', 'female']

    for gender in labels:

        path = paths_celeba['train_' + gender]
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

        random.shuffle(files)

        # move the 15% of each directory files to validation respective directory
        num_15 = int(len(files) * 0.15)

        for i in range(num_15):
            shutil.move(os.path.join(path, files[i]), os.path.join(paths_celeba['valid_' + gender], files[i]))

        print('Validation set: ' + gender + ' images processed')


# This function organizes the dataset, creating train and test folders
# wich contains the training-set and the test.set respectively.
def split_images(src, celeba_path):
    types = ['train', 'test']
    # preparing the dataset
    for type in types:
        print('--- ' + type + ' set pre-processing ---')
        count = 0
        src_path = src + type

        with open("list_attr_celeba_ground.txt") as f:
            for line in f:
                row = line.split(' ')

                name = row[0]
                gender = row[21]

                # gender=-1 => female
                # gender=1 => male
                if gender == "1":
                    dst_img_path = os.path.join(celeba_path[type + '_male'], name)
                else:
                    dst_img_path = os.path.join(celeba_path[type + '_female'], name)

                src_img_path = os.path.join(src_path, name)

                if (count % 10000) == 0:
                    print(count)

                #print(os.path.isfile(src_img_path))

                if os.path.isfile(src_img_path):
                    shutil.copyfile(src_img_path, dst_img_path)
                    count = count + 1

        print("The " + type + " set contain", str(count), " samples")


paths = RepoPaths()
src = os.path.join(paths.root, 'images-dpmcrop-')
# split images in male and female folder for both train and test set
#split_images(src, paths.celeba)
# Create validation set from training set
create_validation(paths.celeba)
# generate the three prototype images
#create_prototypes(paths.celeba, paths.proto)
