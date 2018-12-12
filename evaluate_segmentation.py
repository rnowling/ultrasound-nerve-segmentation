from __future__ import print_function

import argparse

import os
from skimage.measure import label as labelcc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import numpy as np

from data import load_train_data, load_test_data

img_rows = 512
img_cols = 512

smooth = 1.

def np_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = imgs[i]

    imgs_p = imgs_p[..., np.newaxis]

    return imgs_p

def train_and_predict(args):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_mask_test, imgs_flname_test = load_test_data(args.input_dir)
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
    imgs_mask_test = np.around(imgs_mask_test)
    test_labels = np.array([mask_img.flatten().max() > 0 \
                            for mask_img in imgs_mask_test])
    
    pred_masks = np.load(os.path.join(args.input_dir,
                                      'imgs_pred_mask_test.npy'))

        

    pred_masks = pred_masks.astype('float32')
    scaled = pred_masks / 255.
    scaled = np.around(scaled)
    print(set(scaled.flatten()), set(imgs_mask_test.flatten()))

    pred_labels = np.array([mask_img.flatten().max() > 0 \
                            for mask_img in pred_masks])

    dice = np.array([np_dice_coef(imgs_mask_test[i], scaled[i]) \
                     for i in xrange(len(imgs_mask_test))])

    recalls = np.array([recall_score(imgs_mask_test[i].flatten(), scaled[i].flatten()) \
                        for i in xrange(len(imgs_mask_test))])

    pred_components = np.array([labelcc(mask_img, return_num=True)[1]
                                for mask_img in pred_masks])

    if args.omit_empty:
        imgs_mask_test = imgs_mask_test[test_labels]
        pred_masks = pred_masks[test_labels]
        pred_labels = pred_labels[test_labels]
        dice = dice[test_labels]
        recalls = recalls[test_labels]
        pred_components = pred_components[test_labels]
        test_labels = test_labels[test_labels]

    acc = accuracy_score(test_labels,
                         pred_labels)

    recall = recall_score(test_labels,
                          pred_labels)

    print("Dice coefficient:", np.mean(dice), np.std(dice))
    print("Recall per image:", np.mean(recalls), np.std(recalls))
    print("Accuracy:", acc)
    print("Recall:", recall)

    bin_counts = np.bincount(pred_components)
    print("CC distribution:", bin_counts)
    print(">1 CC:", 100. * sum(bin_counts[2:]) / len(pred_masks))

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)

    parser.add_argument("--omit-empty",
                        action="store_true")

    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)
