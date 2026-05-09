#!python

"""
Current Step:
    Computing CIEDE2000 distance mean and std for each skin inler by
    the lookup table made from last step.
    
Last Step:
    Computing CIEDE2000 distances between all RGB pixels and all MSTs.
    Previous plan: Removing gradients from Monk ORBs. But we found it really
    inefficient since we have the swatches. All the orbs are made by the 
    CIELAB color space. Hence, this space is perceptually closer to human
    interpretation of the skin colors.
"""

import sys, os
import pickle
import time
import glob
import numpy as np
import pandas
import multiprocessing as mps
from multiprocessing import shared_memory
import skimage

# globals
br = breakpoint
e = lambda: os._exit(0)
EPSILON = 1e-6
NPS = 35
MST_RGB_255 = [
    [246, 237, 228],
    [243, 231, 219],
    [247, 234, 208],
    [234, 218, 186],
    [215, 189, 150],
    [160, 126,  86],
    [130,  92,  67],
    [ 96,  65,  52],
    [ 58,  49,  42],
    [ 41,  36,  32]
]
XYZ_WHITEPOINT = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
CIEDE_NORMALIZER = 96.0
INLIER_IDIR = "/mnt/isilon/air/data/melanoma/data/inlier/"
RGB_MST_CIEDE2000_IFNAME = "../../data/rgb_mst_ciede2000.pckl"
MEAN_STD_OFNAME = "../../data/inlier_rgb_mst_ciede2000_mu_sig.csv"
    

def lookup_rgb_distances(
    inlier: np.ndarray,
    rgb_mst_ciede: np.ndarray) -> np.ndarray:
    """
    Retrieve precomputed distances for specific RGB colors.

    Parameters
    ----------
    fname : str
        Filename of the image to extract inlier colors from.
    dist_fname : str
        Filename of the precomputed RGB to MST CIEDE2000 distances.
    Returns
    -------
    distances : (N, 16) ndarray
        Distance rows pulled from `rgb_dists` for each color.
    """
    # Allow float colors scaled 0–1
    r, g, b = tuple(inlier[:, c].astype(np.int64) for c in range(3))
    indices = (r << 16) | (g << 8) | b
    return rgb_mst_ciede[indices]

def mu_sig(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation along axis 0."""
    mu = np.mean(data, axis=0)
    sig = np.std(data, axis=0)
    return mu, sig

def _mp(
    fname: str, 
    dist_fname: str=RGB_MST_CIEDE2000_IFNAME) -> tuple[np.ndarray, np.ndarray]:
    # Read the precomputed distances
    with open(dist_fname, mode='rb') as f:
        rgb_mst_ciede = pickle.load(f)
    # Read inlier
    img = skimage.io.imread(fname)[:, :, :3]
    skn = img[:, -256:, :]
    inlier = skn[skn.sum(axis=2) != 0, :]
    dists = lookup_rgb_distances(inlier, rgb_mst_ciede[:, 6:])
    dists /= CIEDE_NORMALIZER
    musig = (np.zeros(10, dtype=np.float64), np.zeros(10, dtype=np.float64))
    if dists.shape[0] > 0:
        musig = mu_sig(dists)
    ratio = len(inlier) / (skn.shape[0] * skn.shape[1])
    result = (ratio, ) + musig
    return result

def main(argv) -> None:    
    # Read files
    fnames = sorted(glob.glob(os.path.join(INLIER_IDIR, "*.png")))
    print(f"Found {len(fnames)} inlier images.")
    # Make the DataFrame to keep the mean and std per MST per image
    df = pandas.DataFrame(
        columns=\
            ["filename", "ratio"] + \
            [f"mst_{i}_mu" for i in range(10)] + \
                [f"mst_{i}_sig" for i in range(10)])
    if NPS == 1:
        for cfn, fn in enumerate(fnames[:10]):
            print(f"Processing {cfn + 1}/{len(fnames)}: {fn}")
            res = _mp(fn)
            df.loc[cfn] = \
                [os.path.basename(fn), res[0]] + \
                list(res[1]) + list(res[2])
    else:
        with mps.Pool(processes=NPS) as pool:
            res = pool.map(_mp, fnames)
            for cfn, (ratio, mu, sig) in enumerate(res):
                df.loc[cfn] = \
                    [os.path.basename(fnames[cfn]), ratio] + \
                    list(mu) + list(sig)
    # Save out mean and std DataFrame
    df.to_csv(MEAN_STD_OFNAME, index=False)
    return None

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
