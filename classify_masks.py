from __future__ import print_function

import argparse

import os
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
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

def validate(args):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    all_true_masks = []
    all_pred_masks = []
    for input_dir in args.folds:
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

    accuracies = np.zeros((len(all_true_masks)))
    recalls = np.zeros((len(all_true_masks)))
    precisions = np.zeros((len(all_true_masks)))
    aucs = []
    for i in xrange(len(all_true_masks)):
        test_pred_masks = all_pred_masks[i]
        test_true_masks = all_true_masks[i]
        remaining_indices = list(set(xrange(len(all_true_masks))) - set([i]))
        train_true_masks = np.concatenate(all_true_masks[remaining_indices])
        train_pred_masks = np.concatenate(all_pred_masks[remaining_indices])

        print(train_true_masks.shape, train_pred_masks.shape)

        probabilities, test_labels = train_and_predict_fold(train_true_masks,
                                                            train_pred_masks,
                                                            test_true_masks,
                                                            test_pred_masks)

        aucs.append(roc_auc_score(test_labels, probabilities))

        fpr, tpr, _ = roc_curve(test_labels, probabilities)
        plt.plot(fpr, tpr)

        binary = np.zeros(probabilities.shape)
        binary[probabilities >= 0.5] = 1.0

        accuracies[i] = accuracy_score(test_labels,
                                       binary)

        recalls[i] = recall_score(test_labels,
                                  binary)

        precisions[i] = precision_score(test_labels,
                                        binary)


    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.savefig("roc_curve_crossfold.png", DPI=300)

    print()
    print("Accuracy mean:", np.mean(accuracies))
    print("Accuracy std:", np.std(accuracies))
    print("Recall mean:", np.mean(recalls))
    print("Recall std:", np.std(recalls))
    print("Precision mean:", np.mean(precisions))
    print("Precision std:", np.std(precisions))
    print("AUC mean:", np.mean(aucs))
    print("AUC std:", np.std(aucs))

def train_and_predict(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    _, test_true_masks, test_imgs_flnames = load_test_data(args.input_dir)
    _, train_true_masks = load_train_data(args.input_dir)

    test_pred_masks = np.load(os.path.join(args.input_dir,
                                           'imgs_pred_mask_test.npy'))
    train_pred_masks = np.load(os.path.join(args.input_dir,
                                            'imgs_pred_mask_train.npy'))

    test_true_masks = test_true_masks.astype('float32')
    test_true_masks /= 255.  # scale masks to [0, 1]
    test_true_masks = np.around(test_true_masks)

    test_pred_masks = test_pred_masks.astype('float32')
    test_pred_masks /= 255.  # scale masks to [0, 1]
    test_pred_masks = np.around(test_pred_masks)

    train_true_masks = train_true_masks.astype('float32')
    train_true_masks /= 255.  # scale masks to [0, 1]
    train_true_masks = np.around(train_true_masks)

    train_pred_masks = train_pred_masks.astype('float32')
    train_pred_masks /= 255.  # scale masks to [0, 1]
    train_pred_masks = np.around(train_pred_masks)

    print(train_true_masks.shape,
          train_pred_masks.shape,
          test_true_masks.shape,
          test_pred_masks.shape)

    probs, test_labels = train_and_predict_fold(train_true_masks,
                                                train_pred_masks,
                                                test_true_masks,
                                                test_pred_masks)

    # pred_labels = np.zeros(probs.shape)
    # pred_labels[probs >= args.threshold] = 1.0

    # accuracy = accuracy_score(test_labels,
    #                           pred_labels)

    # recall = recall_score(test_labels,
    #                       pred_labels)

    # precision = precision_score(test_labels,
    #                             pred_labels)

    # print("Accuracy", accuracy)
    # print("Recall", recall)
    # print("Precision", precision)

    np.save(os.path.join(args.output_dir, 'pred_image_probabilities.npy'),
            probs)

def train_and_predict_fold(train_true_masks, train_pred_masks, test_true_masks, test_pred_masks, output_plots=False):
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

    if output_plots:
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

    #model = RandomForestClassifier(n_estimators = 100)
    model = SGDClassifier(loss="log", penalty="l2", max_iter=10000)
    model.fit(train_features, train_labels)

    print(model.coef_)
    #print(model.feature_importances_)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    probabilities = model.predict_proba(test_features)[:, 1]

    fpr, tpr, thresholds = roc_curve(test_labels, probabilities)
    print("AUC", roc_auc_score(test_labels, probabilities))
    print(thresholds)
    #plt.plot(fpr, tpr)
    #plt.xlabel("FPR", fontsize=16)
    #plt.ylabel("TPR", fontsize=16)
    #plt.savefig("roc_curve.png", DPI=200)

    return probabilities, test_labels

def parseargs():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    classify_parser = subparsers.add_parser("classify",
                                            help="Classify slices")

    classify_parser.add_argument("--input-dir",
                                 type=str,
                                 required=True)

    classify_parser.add_argument("--output-dir",
                                 type=str,
                                 required=True)

    classify_parser.add_argument("--threshold",
                                 type=float,
                                 default=0.15)

    validation_parser = subparsers.add_parser("validate",
                                              help="K-Fold validation")

    validation_parser.add_argument("--folds",
                                   type=str,
                                   nargs="+",
                                   required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()

    if args.mode == "validate":
        validate(args)
    elif args.mode == "classify":
        train_and_predict(args)
    else:
        raise Exception("Unknown mode '%s'" % args.mode)   
