"""
Current Step:
    Separately saving some MST images to take a look
"""

# imports
import sys
import os
import shutil
import time
import numpy as np
import scipy as sp
import pickle


# import multiprocessing as mps

# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
NPS = 1
SPLIT_LEVEL = 1000
TVT_IPCKL = "/mnt/isilon/prarie/experiment/exp_0074/output/tvt.pckl"
INLIER_IDIR = "/mnt/isilon/air/data/melanoma/data/classification_inlier"
MST_ODIR = "../../output/mst_9"
os.makedirs(MST_ODIR, exist_ok=True)

def main(argv):
    with open(TVT_IPCKL, 'rb') as pckl_f:
        df = pickle.load(pckl_f)
    for fn in df[df.mst_9 == 1].combined.tolist():
        bn = os.path.basename(fn)
        ifn = os.path.join(INLIER_IDIR, bn)
        ofn = os.path.join(MST_ODIR, bn)
        shutil.copy2(ifn, ofn)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')