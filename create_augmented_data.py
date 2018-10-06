import argparse
import glob
import os

import numpy as np
from scipy import ndimage
import skimage.io as skio

def read_image_pairs(image_dir, mask_dir):
    for flname in glob.glob(os.path.join(mask_dir, "*.tif")):
        img = skio.imread(flname, as_grey=True)

        basename = os.path.basename(flname)

        path = os.path.join(image_dir, basename)
        mask = skio.imread(path)

        basename = os.path.splitext(basename)[0]
        
        yield img, mask, basename

def image_rotator(triplets):
    for image, mask, basename in triplets:
        for i, suffix in enumerate(["_rot90", "_rot180", "_rot270", "_rot360"]):
            image = np.rot90(image)
            mask = np.rot90(mask)
            yield image, mask, basename + suffix

def image_zoomer(triplets):
    zoom_amount = 2
    for image, mask, basename in triplets:
        image = ndimage.zoom(image, zoom_amount)[256:768, 256:768]
        mask = ndimage.zoom(image, zoom_amount)[256:768, 256:768]
        suffix = "_zoomed2"
        yield image, mask, basename + suffix
        
        
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-img-dir",
                        type=str,
                        required=True)

    parser.add_argument("--input-mask-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-img-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-mask-dir",
                        type=str,
                        required=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    if not os.path.exists(args.output_mask_dir):
        os.makedirs(args.output_mask_dir)
        
    triplets = read_image_pairs(args.input_img_dir, args.input_mask_dir)
    augmented_triplets = image_zoomer(image_rotator(triplets))

    for image, mask, basename in augmented_triplets:
        flname = basename + ".tif"

        print(flname)
        
        image_path = os.path.join(args.output_img_dir,
                                  flname)
        skio.imsave(image_path, image)

        mask_path = os.path.join(args.output_mask_dir,
                                 flname)
        skio.imsave(mask_path, mask)

            

        


