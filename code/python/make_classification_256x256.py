"""
Current Step:
    Separately saving some MST images to take a look
"""

# imports
import sys
import os
import shutil
import glob
import time
import numpy as np
import skimage
import multiprocessing as mps


# import multiprocessing as mps

# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
NPS = 4
SPLIT_LEVEL = 1000
INLIER_IDIR = "/mnt/isilon/air/data/melanoma/data/classification_inlier"
CLASS_256x256 = "/mnt/isilon/air/data/melanoma/data/classification_256x256"

def save_256x256(fn):
    img = skimage.io.imread(fn)
    crop = img[:, :256]
    out_fn = os.path.join(CLASS_256x256, os.path.basename(fn))
    skimage.io.imsave(out_fn, crop)
    pass

def main(argv):
    fns = sorted(glob.glob(os.path.join(INLIER_IDIR, "*.png"),
                           recursive=True))
    print(len(fns))
    if NPS == 1:
        for fn in fns:
            save_256x256(fn)
    else:
        with mps.Pool(processes=NPS) as pool:
            pool.map(save_256x256, fns)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')