from __future__ import print_function

import argparse

import os
from skimage.transform import resize
from skimage.io import imsave
from sklearn.metrics import accuracy_score
import numpy as np

from data import load_train_data, load_test_data

img_rows = 512
img_cols = 512

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]

    return imgs_p

def train_and_predict(args):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_mask_test, imgs_flname_test = load_test_data(args.input_dir)

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
    imgs_mask_test = np.around(imgs_mask_test)
    test_labels = [mask_img.flatten().max() > 0 \
                   for mask_img in imgs_mask_test]
    
    pred_labels = np.load(os.path.join(args.input_dir,
                                       'pred_image_classes.npy'))

    flnames = np.load(os.path.join(args.input_dir,
                                   "imgs_flname_test.npy"))

    
    if args.blacklist:
        with open(args.blacklist) as fl:
            blacklist = set()
            for ln in fl:
                blacklist.add(ln.strip())
        filtered_test_labels = []
        filtered_pred_labels = []
        filtered_flnames = []
        for i in range(len(test_labels)):
            flname = flnames[i]
            if flname not in blacklist:
                filtered_test_labels.append(test_labels[i])
                filtered_pred_labels.append(pred_labels[i])
                filtered_flnames.append(flname)

        test_labels = np.array(filtered_test_labels)
        pred_labels = np.array(filtered_pred_labels)
        flnames = np.array(filtered_flnames)

    acc = accuracy_score(test_labels,
                         pred_labels)

    print("Accuracy:", acc)

    if args.output_mispredictions:
        with open(args.output_mispredictions, "w") as fl:
            fl.write("Filename\tTrue Label\tPredicted Label\n")
            for i in range(len(pred_labels)):
                if pred_labels[i] != test_labels[i]:
                    fl.write(flnames[i])
                    fl.write("\t")
                    fl.write(str(test_labels[i]))
                    fl.write("\t")
                    fl.write(str(pred_labels[i, 0]))
                    fl.write("\n")


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-mispredictions",
                        type=str)

    parser.add_argument("--blacklist",
                        type=str)

    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)
