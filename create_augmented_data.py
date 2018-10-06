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
        # unzoomed image
        yield image, mask, basename

        for i, suffix in enumerate(["_rot90", "_rot180", "_rot270"]):
            image = np.rot90(image)
            mask = np.rot90(mask)
            yield image, mask, basename + suffix

def image_zoomer(triplets):
    zoom_amount = 2
    for image, mask, basename in triplets:
        # unzoomed image
        yield image, mask, basename
        
        image = ndimage.zoom(image, zoom_amount)[256:768, 256:768]
        mask = ndimage.zoom(mask, zoom_amount)[256:768, 256:768]
        suffix = "_zoomed2"
        yield image, mask, basename + suffix

def image_shifter(triplets):
    shift_amount = 64
    shift_multiples = 6
    for image, mask, basename in triplets:
        # unzoomed image
        yield image, mask, basename

        for x_i in xrange(shift_multiples):
            for y_i in xrange(shift_multiples):
                if x_i == 0 and y_i == 0:
                    continue
                
                shifted_image = ndimage.shift(image,
                                              (x_i * shift_amount,
                                               y_i * shift_amount),
                                              mode="reflect")
                shifted_mask = ndimage.shift(mask,
                                             (x_i * shift_amount,
                                              y_i * shift_amount),
                                             mode="reflect")
                suffix = "_xshift" + str(x_i * shift_amount) + "_yshift" + str(y_i * shift_amount)
                
                yield shifted_image, shifted_mask, basename + suffix
        
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

    parser.add_argument("--disable-zoom",
                        action="store_true")

    parser.add_argument("--disable-rotate",
                        action="store_true")

    parser.add_argument("--disable-shift",
                        action="store_true")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    if not os.path.exists(args.output_mask_dir):
        os.makedirs(args.output_mask_dir)
        
    triplets = read_image_pairs(args.input_img_dir, args.input_mask_dir)

    transformations = triplets
    if not args.disable_rotate:
        transformations = image_rotator(triplets)

    if not args.disable_zoom:
        transformations = image_zoomer(transformations)

    if not args.disable_shift:
        transformations = image_shifter(transformations)

    for image, mask, basename in transformations:
        flname = basename + ".tif"

        image_path = os.path.join(args.output_img_dir,
                                  flname)
        skio.imsave(image_path, image)

        mask_path = os.path.join(args.output_mask_dir,
                                 flname)
        skio.imsave(mask_path, mask)

        print(image_path, mask_path)

            

        


