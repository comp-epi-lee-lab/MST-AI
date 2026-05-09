"""
Current Step:
    Adding One more column:
        Starting with MST with lowest number of samples to higher:
                                  (train, valid, test)
            If N < 2000, split by (50%,    20%,  30%)
            Else, split by        (60%,    20%,  20%)

Last Step:
    We extract some information and make plots. The stats are:
        - inlier ratio
        - membership probabilities for each MST
        - Membership rank-1, rank-2, rank-3 for all MSTs

Last Step:
    We use the classification.xlsx files and save the output images in
    original, lesion, frame, skin, inlier
    with size of 256 x (256 x 5).
    
Last Step:
    We use the Boltzmann distribution to compute membership probabilities.
    
Last Step:
    To use Boltzmann distribution, we need to compute the proper tau value.
    Optimal tau for target perplexity 2.5: 0.0183
    
Last Step:
    Computing CIEDE2000 distances between all RGB pixels and all MSTs.
    Previous plan: Removing gradients from Monk ORBs. But we found it really
    inefficient since we have the swatches. All the orbs are made by the 
    CIELAB color space. Hence, this space is perceptually closer to human
    interpretation of the skin colors.
    
    extract
    lesions, frames, skin, and inliers, estimate PDFs using Gaussian Mixture
    Models, and compute membership scores based on KL and L1 distances.
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
from sklearn.model_selection import train_test_split
import sklearn, sklearn.model_selection

# import multiprocessing as mps

# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
NPS = 1
SPLIT_LEVEL = 1000
CLASSIFICATION_IFNAME = "../../data/classification_ranks.xlsx"
CLASSIFICATION_PCKL = "../../output/classification_ranks.pckl"
CLASSIFICATION_IDIR = "/mnt/isilon/air/data/melanoma/data/" \
    "classification_256x256"
TVT_OFNAME = "../../data/tvt.xlsx"
TVT_OPCKL = "../../output/tvt.pckl"

def split_data(df):
    """
    Starting from the category with the lowest number of samples,
    stratify samples into three subsets.
    Then remove the previous samples and do the same for the second
    category with the lowest number of samples.
    Continue until the category with the highest number of samples.
    """
    # Column names are "mst_n"
    # Find the fewest samples column
    mst = df.loc[:, [f"mst_{c+1}" for c in range(10)]].values
    lbs = df["label"].values
    # Check if there is any row in mst which is completely zero:
    nonzero_indices = np.nonzero(mst.sum(axis=1) > 0)[0]
    mst, lbs = mst[nonzero_indices, :], lbs[nonzero_indices]
    # Remaining, Train, Valid, Test: 0, 1, 2, 3
    tvt = np.zeros(mst.shape[0])
    used_indices = []
    while len(used_indices) < lbs.shape[0]:
        minlb_col = mst.sum(axis=0).argmin()
        indices = np.arange(mst.shape[0])[mst[:, minlb_col] == 1].tolist()
        indices = [a for a in indices if not (a in used_indices)]
        used_indices += indices
        lb = lbs[indices]
        # stratified split into train(50%), valid(20%), test(30%)
        if len(indices) > 3:
            tr_idx, temp_idx, y_tr, y_temp = train_test_split(
                indices, lb, train_size=0.5, stratify=lb)
            vl_idx, ts_idx, y_vl, y_ts = train_test_split(
                temp_idx, y_temp, test_size=0.6, stratify=y_temp)
        else:
            # If there are only less than or equal to 3 samples, then put all
            # in the train set.
            tr_idx, vl_idx, ts_idx = indices, [], []
        tvt[tr_idx] = 1
        tvt[vl_idx] = 2
        tvt[ts_idx] = 3
        # Remove the used column
        mst = np.delete(mst, minlb_col, axis=1)
    return tvt

def main(argv):
    if os.path.exists(CLASSIFICATION_PCKL):
        with open(CLASSIFICATION_PCKL, 'rb') as pckl_f:
            df = pickle.load(pckl_f)
    else:
        df = pandas.read_excel(CLASSIFICATION_IFNAME)
        with open(CLASSIFICATION_PCKL, 'wb') as pckl_f:
            pickle.dump(df, pckl_f)
    tvt = split_data(df)
    for x, name in zip([1, 2, 3], ['train', 'valid', 'test']):
        r = (tvt == x).sum() / len(tvt)
        print(f"{name}: {np.round(r, 2)}")
        for mc in range(10):
            mcol = f"mst_{mc+1}"
            bool_indices = (tvt == x) & (df[mcol] == 1)
            lbs = df.loc[bool_indices, "label"].values
            ben, mal = (lbs == 0).sum(), (lbs == 1).sum()
            print(f"\tMST_{mc+1}: benign: {ben}, malignant: {mal}")
    df["combined"] = [os.path.join(CLASSIFICATION_IDIR, os.path.basename(fn))
                      for fn in df["combined"].tolist()]
    df["tvt"] = tvt
    df.to_excel(TVT_OFNAME)
    with open(TVT_OPCKL, mode='wb') as f:
        pickle.dump(df, f)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')