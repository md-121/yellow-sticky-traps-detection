# NOT USED
import os

from matplotlib.pyplot import imshow, show
from patchify import patchify
from matplotlib.image import imread, imsave
import numpy as np
import sys

IMAGE_PATH = "./Datasets_Patched/Images/"
SAVE_DIR = "./plt/"

if __name__ == "__main__":
    for file in os.listdir(IMAGE_PATH):
        if file.endswith(".jpg"):
            print(file)
            image = imread(IMAGE_PATH + file)
            patches = patchify(image, (668, 452, 3), step=1728)
            patches = np.squeeze(patches, axis=2)
            counter = 0
            for x in patches:
                for y in x:
                    imsave(SAVE_DIR + str(counter) + ".jpg", y)
                    counter = counter + 1
            sys.exit()
            print(patches.shape)
            #for patch in patches:
            #    imsave(path, patch)
