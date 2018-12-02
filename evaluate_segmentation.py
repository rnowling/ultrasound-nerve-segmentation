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

smooth = 1.


def np_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


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
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
    test_labels = [mask_img.flatten().max() > 0 \
                   for mask_img in imgs_mask_test]
    
    pred_masks = np.load(os.path.join(args.input_dir,
                                      'imgs_pred_mask_test.npy'))

    pred_masks = pred_masks.astype('float32')
    scaled = pred_masks / 255.

    pred_labels = [mask_img.flatten().max() > 0 \
                   for mask_img in pred_masks]
    
    dice = [np_dice_coef(imgs_mask_test[i],
                         scaled[i]) \
            for i in xrange(len(imgs_mask_test))]

    acc = accuracy_score(test_labels,
                         pred_labels)

    print("Dice coefficient:", np.mean(dice), np.std(dice))
    print("Accuracy:", acc)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)

    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)
