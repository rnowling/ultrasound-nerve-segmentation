import argparse
import glob
import os

import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

def identify_masks(mask_dir):
    for flname in glob.glob(os.path.join(mask_dir, "*.tif")):
        img = imread(flname, as_grey=True)

        has_mask = img.flatten().max() > 0
        yield os.path.basename(flname), img, has_mask

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

    parser.add_argument("--include-empty-masks",
                        type=str,
                        default=None,
                        choices=["testing-set",
                                 "all"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    images = []
    masks = []
    no_mask_images = []
    no_mask_masks = []
    flnames = []
    no_mask_flnames = []
    for flname, mask_img, has_mask in identify_masks(args.mask_dir):
        path = os.path.join(args.image_dir,
                            flname)
        img = imread(path)

        basename = os.path.splitext(flname)[0]

        if has_mask:
            images.append(img)
            masks.append(mask_img)
            flnames.append(basename)
        elif args.include_empty_masks:
            no_mask_images.append(img)
            no_mask_masks.append(mask_img)
            no_mask_flnames.append(basename)

    septlet = train_test_split(images, masks, flnames, test_size = 0.333)
    train_images, test_images, train_masks, test_masks, train_flnames, test_flnames = septlet

    if args.include_empty_masks == "testing-set":
        test_images.extend(no_mask_images)
        test_masks.extend(no_mask_masks)
        test_flnames.extend(no_mask_flnames)
    else:
        septlet = train_test_split(no_mask_images, no_mask_masks, no_mask_flnames, test_size = 0.333)
        no_mask_train_images, no_mask_test_images, no_mask_train_masks, no_mask_test_masks, no_mask_train_flnames, no_mask_test_flnames = septlet
        train_images.extend(no_mask_train_images)
        test_images.extend(no_mask_test_images)
        train_masks.extend(no_mask_train_masks)
        test_masks.extend(no_mask_test_masks)
        train_flnames.extend(no_mask_train_flnames)
        test_flnames.extend(no_mask_test_flnames)

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

        

    
