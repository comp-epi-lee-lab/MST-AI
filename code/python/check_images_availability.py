"""
Current Step:
    Check if the images path are correctly set.
"""

# imports
import sys
import os
import time
import glob
import numpy as np
import scipy as sp
import pickle
import pandas
import skimage

# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
NPS = 38
ISIZE = 256
CLASSIFICATION_IFNAME = \
    "/mnt/isilon/air/data/melanoma/list/classification.xlsx"

def main(argv):
    df = pandas.read_excel(CLASSIFICATION_IFNAME)
    fnames = df['image'].values.tolist()
    print(len(fnames))
    for i, fname in enumerate(fnames):
        if not os.path.isfile(fname):
            print(fname)
            br()
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')