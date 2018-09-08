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

    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    parser.add_argument("--training-filter",
                        type=str,
                        default="no-filtering",
                        choices=["no-filtering",
                                 "patients-without-tumors",
                                 "empty-masks"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    patient_groups = defaultdict(list)
    for flname, mask_img in identify_masks(args.mask_dir):
        path = os.path.join(args.image_dir,
                            flname)
        img = imread(path)

        basename = os.path.splitext(flname)[0]
        patient_id = basename[:basename.find("_", 4)]
        patient_groups[patient_id].append((img, mask_img, flname))

    patient_has_tumor = []
    patient_groups = list(patient_groups.values())
    for triplets in patient_groups:
        has_tumor = any([mask_img.flatten().max() > 0 \
                         for _, mask_img, _ in triplets])
        patient_has_tumor.append(has_tumor)

    quadlet = train_test_split(patient_groups,
                               patient_has_tumor,
                               test_size = 0.333,
                               stratify = patient_has_tumor)
    train_groups, test_groups, train_tumors, test_tumors =  quadlet

    if args.training_filter == "patients-without-tumors":
        print("Filtering patients without tumors out of training set")
        print(len(train_groups))
        train_groups = [t for (t, has_tumor) in zip(train_groups, train_tumors) \
                        if has_tumor]
        print(len(train_groups))

    train_triplet = chain(*train_groups)
    test_triplet = chain(*test_groups)

    if args.training_filter == "empty-masks":
        train_triplet = [t for t in train_triplet \
                         if t[1].flatten().max() > 0]

    train_images, train_masks, train_flnames = zip(*train_triplet)
    test_images, test_masks, test_flnames = zip(*test_triplet)

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

        

    
