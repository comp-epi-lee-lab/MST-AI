#!python

"""
Current Step:
    To use Boltzmann distribution, we need to compute the proper tau value.
    Optimal tau for target perplexity 2.5: 0.0183
    
Last Step:
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
import scipy as sp
import pandas
import multiprocessing as mps
import skimage

# globals
br = breakpoint
e = lambda: os._exit(0)
EPSILON = 1e-6
NPS = 1
MEAN_STD_IFNAME = "../../data/inlier_rgb_mst_ciede2000_mu_sig.csv"

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
    probs = w / (w.sum() + EPSILON)
    return probs

def entropy(p, eps=1e-12):
    """
    Compute the entropy of a probability distribution.
    """
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum()

def tune_tau_by_entropy(mus_list, sigmas_list, taus, target_ppx=2.5):
    """
    Tune tau to achieve target perplexity by minimizing entropy difference.
    Args:
        mus_list: list of arrays of length K
        sigmas_list: list of arrays of length K
        taus: array of tau values to evaluate
        target_ppx: target perplexity
    Returns:
        tau_star: optimal tau value
        H_mean: mean entropy across all inputs for each tau
        ppx_mean: mean perplexity across all inputs for each tau
    """
    H_all = []
    for mu, sigma in zip(mus_list, sigmas_list):          # each is (K,)
        Eabs = expected_abs(np.asarray(mu), np.asarray(sigma))
        H = []
        for t in taus:
            p = probs_from_Eabs(Eabs, t)
            H.append(entropy(p))
        H_all.append(H)
    H_mean = np.mean(H_all, axis=0)                        # (T,)
    ppx_mean = np.exp(H_mean)
    tau_star = taus[np.argmin(np.abs(ppx_mean - target_ppx))]
    return tau_star, H_mean, ppx_mean
    
def main(argv) -> None:    
    musigs = pandas.read_csv(MEAN_STD_IFNAME)
    mus_col_names = [f"mst_{k}_mu" for k in range(10)]
    sig_col_names = [f"mst_{k}_sig" for k in range(10)]
    mus_list = [musigs.loc[i, mus_col_names].values.astype(np.float64)
                for i in range(musigs.shape[0])]
    sigmas_list = [musigs.loc[i, sig_col_names].values.astype(np.float64)
                   for i in range(musigs.shape[0])]
    taus = np.linspace(0.0001, 1.0000, 10000)                # sweep
    target_ppx = 2.5
    tau_star, H_mean, ppx_mean = tune_tau_by_entropy(
        mus_list, sigmas_list, taus, target_ppx=target_ppx)
    print(f"Optimal tau for target perplexity {target_ppx}: {tau_star:.4f}")    
    return None

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
