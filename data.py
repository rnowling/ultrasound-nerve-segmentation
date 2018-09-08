import os

import numpy as np

def load_train_data(dirname):
    imgs_train = np.load(os.path.join(dirname, 'imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(dirname, 'imgs_mask_train.npy'))
    return imgs_train, imgs_mask_train

def load_test_data(dirname):
    imgs_test = np.load(os.path.join(dirname, 'imgs_test.npy'))
    imgs_mask_test = np.load(os.path.join(dirname, 'imgs_mask_test.npy'))
    imgs_flname_test = np.load(os.path.join(dirname, 'imgs_flname_test.npy'))
    return imgs_test, imgs_mask_test, imgs_flname_test
