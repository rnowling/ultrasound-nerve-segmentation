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

    parser.add_argument("--test-mask-dir",
                        type=str,
                        required=True)

    parser.add_argument("--test-image-dir",
                        type=str,
                        required=True)

    parser.add_argument("--train-mask-dir",
                        type=str,
                        required=True)

    parser.add_argument("--train-image-dir",
                        type=str,
                        required=True)
    
    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    test_triplet = []
    for flname, mask_img in identify_masks(args.test_mask_dir):
        path = os.path.join(args.test_image_dir,
                            flname)
        img = imread(path)
        test_triplet.append((img, mask_img, flname))

    train_triplet = []
    for flname, mask_img in identify_masks(args.train_mask_dir):
        path = os.path.join(args.train_image_dir,
                            flname)
        img = imread(path)

        train_triplet.append((img, mask_img, flname))

    test_images, test_masks, test_flnames = zip(*test_triplet)
    train_images, train_masks, train_flnames = zip(*train_triplet)
        
    print(len(train_images), "training images")
    print(len(test_images), "testing images")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, "imgs_train.npy"),
            np.array(train_images))

    np.save(os.path.join(args.output_dir, "imgs_mask_train.npy"),
            np.array(train_masks))

    np.save(os.path.join(args.output_dir, "imgs_test.npy"),
            np.array(test_images))

    np.save(os.path.join(args.output_dir, "imgs_mask_test.npy"),
            np.array(test_masks))

    np.save(os.path.join(args.output_dir, "imgs_flname_train.npy"),
            np.array(train_flnames))

    np.save(os.path.join(args.output_dir, "imgs_flname_test.npy"),
            np.array(test_flnames))

        

    
