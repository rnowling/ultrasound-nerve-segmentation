import argparse
import glob
import os

import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

def identify_masks(mask_dir):
    for flname in glob.glob(os.path.join(mask_dir, "*.tif")):
        img = imread(flname, as_grey=True)

        if img.flatten().max() > 0:
            yield os.path.basename(flname), img

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask-dir",
                        type=str,
                        required=True)

    parser.add_argument("--image-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    images = []
    masks = []
    for flname, mask_img in identify_masks(args.mask_dir):
        path = os.path.join(args.image_dir,
                            flname)
        img = imread(path)
        images.append(img)
        masks.append(mask_img)

    quadlet = train_test_split(images, masks, test_size = 0.333)
    train_images, test_images, train_masks, test_masks = quadlet

    print(len(train_images), "training images")
    print(len(test_images), "testing images")

    np.save(os.path.join(args.output_dir, "imgs_train.npy"),
            np.array(train_images))

    np.save(os.path.join(args.output_dir, "imgs_mask_train.npy"),
            np.array(train_masks))

    np.save(os.path.join(args.output_dir, "imgs_test.npy"),
            np.array(test_images))

    np.save(os.path.join(args.output_dir, "imgs_id_test.npy"),
            np.array(test_masks))

        

    
