import argparse
from collections import defaultdict
import glob
from itertools import chain
import shutil
import os

import numpy as np
from skimage.io import imread
from sklearn.model_selection import KFold

def find_pairs(image_dir, mask_dir):
    for mask_path in glob.glob(os.path.join(mask_dir, "*.tif")):
        mask_basename = os.path.basename(mask_path)

        img_path = os.path.join(image_dir, mask_basename)
        
        yield mask_path, img_path, mask_basename

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask-dir",
                        type=str,
                        required=True)

    parser.add_argument("--image-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-prefix",
                        type=str,
                        required=True)

    parser.add_argument("--training-filter",
                        type=str,
                        default="no-filtering",
                        choices=["no-filtering",
                                 "empty-masks"])

    parser.add_argument("--num-splits",
                        type=int,
                        default=5)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    patient_groups = defaultdict(list)
    for mask_flname, img_flname, flname in find_pairs(args.image_dir, args.mask_dir):
        if args.training_filter == "empty-masks":
            mask_img = imread(mask_flname, as_grey=True)
            if mask_img.flatten().max() == 0:
                continue

        patient_id = flname[:flname.find("_", 4)]
        
        patient_groups[patient_id].append((img_flname, mask_flname, flname))

    print("Found", len(patient_groups), "patients")

    patient_triplets = np.array(list(patient_groups.values()))
    X = np.zeros((len(patient_triplets), 2))
    kf = KFold(n_splits=args.num_splits,
               shuffle=True)
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        train_patients = patient_triplets[train_index]
        test_patients = patient_triplets[test_index]
        train_triplets = list(chain(*train_patients))
        test_triplets = list(chain(*test_patients))

        print(len(train_triplets), "training patients")
        print(len(test_triplets), "testing patients")

        output_dir = args.output_prefix + str(i)

        train_masks_dir = os.path.join(output_dir,
                                       "train",
                                       "masks")
        
        train_images_dir = os.path.join(output_dir,
                                        "train",
                                        "images")

        test_masks_dir = os.path.join(output_dir,
                                      "test",
                                      "masks")
        
        test_images_dir = os.path.join(output_dir,
                                       "test",
                                       "images")

        dirnames = [train_masks_dir,
                    train_images_dir,
                    test_masks_dir,
                    test_images_dir]
        for dirname in dirnames:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        for img_flname, mask_flname, flname in train_triplets:
            shutil.copy(img_flname,
                        os.path.join(train_images_dir, flname))
            shutil.copy(mask_flname,
                        os.path.join(train_masks_dir, flname))

        for img_flname, mask_flname, flname in test_triplets:
            shutil.copy(img_flname,
                        os.path.join(test_images_dir, flname))
            shutil.copy(mask_flname,
                        os.path.join(test_masks_dir, flname))


        

    
