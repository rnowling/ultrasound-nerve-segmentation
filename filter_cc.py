import argparse
import os

import numpy as np
from skimage.io import imsave
from skimage.measure import label as labelcc

img_rows = 512
img_cols = 512

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)
    
    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    parser.add_argument("--classification-dir",
                        type=str,
                        required=True)

    parser.add_argument("--min-area",
                        type=float,
                        default=0.025)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred_dir = os.path.join(args.output_dir,
                            "preds")
        
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    test_images = np.load(os.path.join(args.input_dir,
                                       "imgs_test.npy"))

    np.save(os.path.join(args.output_dir, "imgs_test.npy"),
            test_images)

    test_masks = np.load(os.path.join(args.input_dir,
                                       "imgs_mask_test.npy"))

    np.save(os.path.join(args.output_dir, "imgs_mask_test.npy"),
            test_masks)
        
    test_flnames = np.load(os.path.join(args.input_dir,
                                        "imgs_flname_test.npy"))

    np.save(os.path.join(args.output_dir, "imgs_flname_test.npy"),
            test_flnames)


    pred_masks = np.load(os.path.join(args.input_dir,
                                      "imgs_pred_mask_test.npy"))

    pred_labels = np.load(os.path.join(args.classification_dir,
                                       "pred_image_classes.npy"))

    empty_mask = np.zeros((img_rows, img_cols, 1))
    for i, (image_id, mask, label) in enumerate(zip(test_flnames, pred_masks, pred_labels)):

        if label == 0:
            pred_masks[i] = empty_mask
        else:
            cc_labels, num_ccs = labelcc(mask, return_num=True)

            for j in xrange(1, num_ccs):
                area = float(mask[cc_labels == j].size) / mask.size
                print j, area
                if area < args.min_area:
                    mask[cc_labels == j] = 0

        pred_masks[i] = mask

        imsave(os.path.join(pred_dir,
                            str(image_id[:-4]) + '_pred.png'),
               mask[:, :, 0])

    np.save(os.path.join(args.output_dir, "imgs_pred_mask_test.npy"),
            pred_masks)
