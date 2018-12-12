from __future__ import print_function

import argparse

import os
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from skimage.measure import label as labelcc

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

from data import load_train_data, load_test_data

img_rows = 512
img_cols = 512


def train_and_predict(args):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    all_true_masks = []
    all_pred_masks = []
    for input_dir in args.input_dir:
        _, true_masks, imgs_flnames = load_test_data(input_dir)
        pred_masks = np.load(os.path.join(input_dir,
                                          'imgs_pred_mask_test.npy'))

        true_masks = true_masks.astype('float32')
        true_masks /= 255.  # scale masks to [0, 1]
        true_masks = np.around(true_masks)

        pred_masks = pred_masks.astype('float32')
        pred_masks /= 255.  # scale masks to [0, 1]
        pred_masks = np.around(pred_masks)
        
        all_true_masks.append(true_masks)
        all_pred_masks.append(pred_masks)

    all_true_masks = np.array(all_true_masks)
    all_pred_masks = np.array(all_pred_masks)

    accuracies = []
    for i in xrange(len(all_true_masks)):
        test_pred_masks = all_pred_masks[i]
        test_true_masks = all_true_masks[i]
        remaining_indices = list(set(xrange(len(all_true_masks))) - set([i]))
        train_true_masks = np.concatenate(all_true_masks[remaining_indices])
        train_pred_masks = np.concatenate(all_pred_masks[remaining_indices])

        print(train_true_masks.shape, train_pred_masks.shape)

        acc =train_and_predict_fold(train_true_masks,
                                    train_pred_masks,
                                    test_true_masks,
                                    test_pred_masks)
        accuracies.append(acc)

    print()
    print("Accuracy mean:", np.mean(accuracies))
    print("Accuracy std:", np.std(accuracies))

def train_and_predict_fold(train_true_masks, train_pred_masks, test_true_masks, test_pred_masks):
    train_labels = np.array([1.0 if mask_img.flatten().max() > 0 else 0.0 \
                             for mask_img in train_true_masks])

    test_labels = np.array([1.0 if mask_img.flatten().max() > 0 else 0.0 \
                            for mask_img in test_true_masks])

    train_pred_components = np.array([labelcc(mask_img, return_num=True)[1]
                                      for mask_img in train_pred_masks]).reshape(-1, 1)

    test_pred_components = np.array([labelcc(mask_img, return_num=True)[1]
                                     for mask_img in test_pred_masks]).reshape(-1, 1)

    train_pred_areas = np.array([np.sum(mask_img)
                                 for mask_img in train_pred_masks]).reshape(-1, 1)

    test_pred_areas = np.array([np.sum(mask_img)
                                 for mask_img in test_pred_masks]).reshape(-1, 1)
    
    train_pred_masks = train_pred_masks.reshape(-1, img_rows * img_cols)
    test_pred_masks = test_pred_masks.reshape(-1, img_rows * img_cols)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    positives_train = train_labels == 1.0
    negatives_train = train_labels == 0.0

    umap_ = umap.UMAP(n_neighbors=10, n_components=5)
    projected = umap_.fit_transform(train_true_masks.reshape(-1, img_rows * img_cols))
    projected_train = umap_.transform(train_pred_masks)
    projected_test = umap_.transform(test_pred_masks)

    plt.clf()
    plt.scatter(projected[negatives_train, 0],
                projected[negatives_train, 1],
                label="N")
    plt.scatter(projected[positives_train, 0],
                projected[positives_train, 1],
                label="P")
    plt.legend()
    plt.savefig("projected_umap_masks.png", DPI=300)
    
    plt.clf()
    plt.scatter(projected_train[negatives_train, 0],
                projected_train[negatives_train, 1],
                label="N")
    plt.scatter(projected_train[positives_train, 0],
                projected_train[positives_train, 1],
                label="P")
    plt.legend()
    plt.savefig("projected_umap_pred_masks.png", DPI=300)

    train_features = np.hstack([projected_train,
                                train_pred_areas])

    test_features = np.hstack([projected_test,
                               test_pred_areas])

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    #model = RandomForestClassifier(n_estimators = 250)
    model = SGDClassifier(loss="log", penalty="l2", max_iter=5000)
    model.fit(train_features, train_labels)

    print(model.coef_)
    #print(model.feature_importances_)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    binary = model.predict(test_features)

    accuracy = accuracy_score(test_labels,
                              binary)

    print("Accuracy:", accuracy)

    return accuracy

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        type=str,
                        nargs="+",
                        required=True)

    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)
