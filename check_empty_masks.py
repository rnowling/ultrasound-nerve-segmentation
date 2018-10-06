import argparse
from collections import defaultdict
import glob
from itertools import chain
import os

import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

def identify_masks(mask_dir):
    for flname in glob.glob(os.path.join(mask_dir, "*.tif")):
        img = imread(flname, as_grey=True)

        basename = os.path.basename(flname)

        yield basename, img

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask-dir",
                        type=str,
                        required=True)

    parser.add_argument("--image-dir",
                        type=str,
                        required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    triplet = []
    for flname, mask_img in identify_masks(args.mask_dir):
        path = os.path.join(args.image_dir,
                            flname)
        img = imread(path)
        triplet.append((img, mask_img, flname))

    total_images = len(triplet)
    prostate_images = sum([1 for t in triplet \
                           if t[1].flatten().max() > 0])

    print(total_images, "images")
    print(prostate_images, " of the images have prostates")

        

    
