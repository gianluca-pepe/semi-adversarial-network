'''
This script has been created by Bruno Marino and Gianluca Pepe
'''
'''
In this file we generate female, male and neutral prototypes using images from training-set
'''

import os, numpy, PIL
from PIL import Image


def create_prototypes(src, dst):
    print('--- Generating prototypes ---')
    types = ['male', 'female', 'neutral']

    for label in types:
        print(label + ' prototype...')
        if label == 'neutral':
            create_neutral_prototype(dst)
        else:
            # Access all JPG files in directory
            imgpath = src['train_'+label]
            allfiles = os.listdir(imgpath)
            imlist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]

            # Assuming all images are the same size, get dimensions of first image
            w, h = Image.open(os.path.join(imgpath, imlist[0])).size
            N = len(imlist)

            # Create a numpy array of floats to store the average (assume RGB images)
            arr = numpy.zeros((h, w, 3), numpy.float)

            # Build up average pixel intensities, casting each image as an array of floats
            for im in imlist:
                imarr = numpy.array(Image.open(os.path.join(imgpath, im)), dtype=numpy.float)
                arr = arr + imarr / N

            # Round values in array and cast as 8-bit integer
            arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)

            # Generate, save and preview final image
            out = Image.fromarray(arr, mode="RGB")
            out.save(os.path.join(dst, label+'.jpg'))
            out.show()

        print(label + ' prototype generated')



def create_neutral_prototype(dst):
    imgf = os.path.join(dst, "female.jpg")
    imgm = os.path.join(dst, "male.jpg")

    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(imgf).size

    # number of pictures in training set
    N = 150724

    # number of male in training set
    NM = 62517

    # number of female in training set
    NF = 88207

    # male proportion in training set
    alpham = NM / N

    # female proportion in training set
    alphaf = NF / N

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = numpy.zeros((h, w, 3), numpy.float)

    # Build up weighted average pixel intensities, casting each image as an array of floats
    imarrm = numpy.array(Image.open(imgm), dtype=numpy.float)
    imarrf = numpy.array(Image.open(imgf), dtype=numpy.float)

    arr = (alpham * imarrm) + (alphaf * imarrf)

    # Round values in array and cast as 8-bit integer
    arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save(os.path.join(dst, 'neutral.jpg'))
    out.show()
