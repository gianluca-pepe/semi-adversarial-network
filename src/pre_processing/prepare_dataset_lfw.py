'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
"""

Prepare LFW dataset, we want this final organization

LFW
—------male
—----------------OR_1
—----------------OR_2
—----------------SM
—----------------NT
—----------------NT
—------female
—----------------OR_1
—----------------OR_2
—----------------SM
—----------------NT
—----------------NT


In this file we create OR_1 and OR_2
"""

import os
from shutil import copyfile
from repo_paths import RepoPaths


GENDERS = ["female", "male"]

# os.getcwd() get current directory

paths = RepoPaths()

src = os.path.join(paths.root, 'lfw-deepfunneled')

count = 0
for gender in GENDERS:
    # take the correct ground truth file both for females and male
    ground_truth = os.path.join(paths.root, gender + "_names.txt")

    with open(ground_truth) as f:
        for line in f:
            # take the name without _000x.jpg
            name = line[:-10]
            numpic = line[-9:-5]

            # print (name + " " + numpic)
            # print(numpic)
            if numpic == "0002":
                # Copy img in the right folder
                src1 = os.path.join(src, name, name + '_0001.jpg')
                src2 = os.path.join(src, name, name + '_0002.jpg')
                dst1 = os.path.join(paths.lfw[gender]['or1'], name + ".jpg")
                dst2 = os.path.join(paths.lfw[gender]['or2'], name + ".jpg")

                if os.path.isfile(src1):
                    copyfile(src1, dst1)
                if os.path.isfile(src2):
                    copyfile(src2, dst2)

print("Dataset prepared!")
