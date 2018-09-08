import argparse
from collections import defaultdict
import glob
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

    parser.add_argument("--empty-masks",
                        type=str,
                        default="exclude-all",
                        choices=["exclude-all",
                                 "testing-set",
                                 "include-all"])

    parser.add_argument("--partitioning",
                        type=str,
                        default="random",
                        choices=["random",
                                 "by-patient"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    triplets = []
    for flname, mask_img in identify_masks(args.mask_dir):
        path = os.path.join(args.image_dir,
                            flname)
        img = imread(path)

        basename = os.path.splitext(flname)[0]

        triplets.append((img, mask_img, basename))

    if args.partitioning == "random":
        has_tumor = [(1 if mask_img.flatten().max() > 0 else 0) \
                     for _, mask_img, _ in triplets]
        train_triplet, test_triplet = train_test_split(triplets,
                                                       test_size = 0.333,
                                                       stratify = has_tumor)
    """
    elif args.partitioning == "by-patient":
        patient_groups = defaultdict(list)

        for img, mask_img, flname in triplets:
            patient_id = basename[:basename.find("_", 4)]
            patient_groups[patient_id].append(img, mask_img, flname)

        patient_has_tumor = []
        for patient_id, triplets in patient_has_tumor.items():
    """

    if args.empty_masks == "testing-set":
        train_triplet = [t for t in train_triplet \
                         if t[1].flatten().max() > 0]
    elif args.empty_masks == "exclude-all":
        train_triplet = [t for t in train_triplet \
                         if t[1].flatten().max() > 0]
        test_triplet = [t for t in test_triplet \
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

        

    
