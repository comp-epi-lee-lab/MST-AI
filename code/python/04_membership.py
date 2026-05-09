#!python

"""
Current Step:
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
"""

import sys, os
import pickle
import time
import glob
import numpy as np
import scipy as sp
import pandas
import multiprocessing as mps
import skimage

# globals
br = breakpoint
e = lambda: os._exit(0)
EPSILON = 1e-6
NPS = 38
TAU = 0.0183
MEAN_STD_IFNAME = "../../data/inlier_rgb_mst_ciede2000_mu_sig.csv"
MEMBERSHIP_OFNAME = "../../data/inlier_rgb_membership_probs.csv"

def expected_abs(mu, sigma):
    """
    Compute E[|X|] where X ~ N(mu, sigma^2)
    """
    a = mu / (sigma + EPSILON)
    eabs = sigma * np.sqrt(2/np.pi) * np.exp(-0.5*a*a) + \
        mu * (2*sp.stats.norm.cdf(a) - 1)
    return eabs

def probs_from_Eabs(Eabs, tau):
    """
    Compute Boltzmann probabilities from expected absolute distances.
    """
    w = np.exp(-Eabs / tau)
    return w / w.sum()

def membership_probs(mu, sigma, tau=TAU):
    """
    Compute membership probabilities using Boltzmann distribution.
    Args:
        mu: array of length K, mean distances to K MSTs
        sigma: array of length K, stddev of distances to K MSTs
        tau: temperature parameter
    Returns:
        probs: array of length K, membership probabilities
    """
    Eabs = expected_abs(np.asarray(mu), np.asarray(sigma))
    probs = probs_from_Eabs(Eabs, tau)
    return probs

def main(argv) -> None:    
    df = pandas.read_csv(MEAN_STD_IFNAME)
    mus_col_names = [f"mst_{k}_mu" for k in range(10)]
    sig_col_names = [f"mst_{k}_sig" for k in range(10)]
    mus_list = [df.loc[i, mus_col_names].values.astype(np.float64)
                for i in range(df.shape[0])]
    sigmas_list = [df.loc[i, sig_col_names].values.astype(np.float64)
                   for i in range(df.shape[0])]
    membership_probs_list = []
    if NPS == 1:
        for mu, sigma in zip(mus_list[:10], sigmas_list[:10]):
            p = membership_probs(mu, sigma, tau=TAU)
            membership_probs_list.append(p)
    else:
        with mps.Pool(NPS) as pool:
            args = [(mu, sigma, TAU) for mu, sigma in zip(mus_list, sigmas_list)]
            membership_probs_list = pool.starmap(membership_probs, args)
    for k in range(10):
        df[f"mst_{k}_prob"] = [membership_probs_list[c][k] 
                               for c in range(df.shape[0])]
    df.to_csv(MEMBERSHIP_OFNAME, index=False)
    return None

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
