"""
MST-AI: Monk Skin Lesion AI
Author: VK
Date: 2025-08-20
Version: 1.0
Description:
    This script implements the MST-AI pipeline for skin lesion analysis.
    It includes functions to load and process Monk ORB images, extract
    lesions, frames, skin, and inliers, estimate PDFs using Gaussian Mixture
    Models, and compute membership scores based on KL and L1 distances.
"""

# imports
import sys
import os
import time
import glob
import scipy as sp
import numpy as np

import pickle
import skimage
import sklearn, sklearn.mixture, sklearn.ensemble, sklearn.exceptions
import torch, torchvision


import warnings

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.ConvergenceWarning)
# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
MODEL_IDIR = "/jewelry1/members/stheanraj/mst/webapp/trial_0000.pckl"
EXAMPLE_IFNAME = "./example/0_1_1_0390.png"


# EXAMPLE_IFNAME = "./example/0_0_0_0002.png"

class MSTAI:
    """
    Class to perform all the steps of the MST-AI pipeline.
    Attributes:
        msts_idir: String, directory containing ORB images.
        msts: List of numpy arrays, ORB samples.
        msts_pdfs: List of fitted Gaussian Mixture Models for ORBs.
        transforms: torchvision.transforms.Compose, image transformations.
        model: Loaded pre-trained model for lesion extraction.
    Methods:
        make_transforms: Create image transformations.
        load_model: Load a pre-trained model from a file.
        get_lesion: Extract lesion pixels from the image.
        get_frame: Extract frame pixels from the image.
        get_skin: Extract skin pixels from the image.
        get_inliers: Extract inlier pixels from the image.
        get_monk_pixels: Load and process ORB images.
        get_pdf: Estimate the PDF of a sample using GMM.
        get_pdf_vals: Get PDF values for a range of points in 3D space.
        get_kl_distances: Compute KL distances between ORB PDFs and image PDF.
        get_l1_distances: Compute L1 distances between ORB PDFs and image PDF.
        get_membership_score: Compute membership scores from distances.
    """

    def __init__(self, msts_idir: str):
        """
        Initialize the MSTAI class with the directory of Monk orbs.
            msts_idir: String, directory containing ORB images.
        """
        self.msts_idir = msts_idir
        # The following attributes will be set once the methods are called
        self.msts = None
        self.msts_pdfs = None
        self.transforms = None
        self.model = None

    def make_transforms(self, isize: int = 256):
        """
        Create a list of transformations for image processing.
            isize: Integer, size of the image to be transformed.
        Returns a list of transformation functions.
        """
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(isize, isize)),
                torchvision.transforms.Normalize([0], [1])
            ])
        self.transforms = transforms
        return transforms

    def load_model(self, model_fn: str):
        """
        Load a pre-trained model from a file.
            model_fn: String, path to the model file.
        Returns the loaded model.
        """
        with open(model_fn, mode='rb') as f:
            model = pickle.load(f)
        model = list(model['bests'].values())[0]['model']
        self.model = model
        return model

    def get_lesion(self, img: np.array):
        """
        Extract lesion pixels from the image.
            img: Numpy array, input image.
        Returns a binary numpy array (binary mask) of lesion pixels.
        """
        # Normalize the image to [0, 1]
        x = (img - img.min()) / (img.max() - img.min() + EPS)
        if self.transforms is None:
            self.transforms = self.make_transforms(isize=256)
        x = self.transforms(x)
        # Add batch dimension and convert to float tensor
        x = x.unsqueeze(dim=0).float()
        if self.model is None:
            self.model = self.load_model(MODEL_IDIR)
        out = self.model(x)
        out = out['out'].sigmoid().squeeze().squeeze().detach().numpy()
        # Thresholding to create binary mask
        lesion = 1 * (out > 0.5)
        return lesion

    def get_frame(self, img: np.array, border_threshold: float = 0.05):
        """
        Extract frame pixels from the image.
        This function is the same as get_convex_hull (3.5) in exp_005.
            img: Numpy array, input image.
        Returns a binary numpy array (binary mask) of frame pixels.
        """
        gray = skimage.color.rgb2gray(img)
        scale = lambda x: (x - x.min()) / (x.max() - x.min() + EPS)
        hc = scale(gray)
        hc = 0.5 * (1 + np.tanh(4 * hc - 2))
        bin = 1.0 * (hc > border_threshold)
        # Convex Hull
        chull = skimage.morphology.convex_hull_image(bin)
        # If there is no convex hull, maybe the border is white, so the binary
        # image should be inverted
        if np.all(chull == 1):
            bin = 1.0 * (hc < (1 - border_threshold))
            chull = skimage.morphology.convex_hull_image(bin)
        frame = 1 - chull
        return frame

    def get_skin(self, img: np.array,
                 lesion: np.array = None, frame: np.array = None):
        """
        Extract skin pixels from the image.
        This function must remove the lesion and frames.
            img: Numpy array, input image.
        Returns a numpy array of skin pixels.
        """
        lesion = lesion if lesion is not None else np.zeros_like(img[:, :, 0])
        # skimage.io.imsave('lesion.png', lesion.astype(np.uint8) * 255)
        frame = frame if frame is not None else np.zeros_like(img[:, :, 0])
        # skimage.io.imsave('frame.png', frame.astype(np.uint8) * 255)
        lesion_frame_less = (1 - lesion) * (1 - frame)
        lesion_frame_less = lesion_frame_less[:, :,
                            np.newaxis].repeat(3, axis=2)
        # skimage.io.imsave('lesion_frame_less.png',
        #                   lesion_frame_less.astype(np.uint8) * 255)
        # Extract skin pixels
        skin = img * lesion_frame_less
        # skimage.io.imsave('skin.png', skin.astype(np.uint8))
        return skin

    def get_inliers(self, skin: np.array):
        """
        Extract inlier pixels from the image.
            img: Numpy array, input image, removed lesion and frame.
        Returns a numpy array of inlier pixels.
        """
        inlier = skin.copy()
        y = skin[skin.sum(axis=2) != 0, :]
        if y.shape[0] != 0:
            # model = sklearn.linear_model.SGDOneClassSVM(nu=0.15)
            model = sklearn.ensemble.IsolationForest(n_estimators=50)
            # model = sklearn.neighbors.LocalOutlierFactor(n_neighbors=20)
            # lbs = model.fit_predict(skin.reshape((-1, 3)))
            model.fit(y)
            lbs = model.predict(skin.reshape((-1, 3)))
            print(np.unique(lbs), (lbs == -1).sum(), (lbs == 1).sum())
            # Reconstruct the image with black color for outliers
            inlier = np.copy(skin)
            inlier[lbs.reshape(skin.shape[:2]) == -1, :] = np.array([0, 0, 0])
        inlier = inlier.astype(np.uint8)
        # skimage.io.imsave('inlier.png', inlier)
        return inlier

    def get_monk_pixels(self, msts_idir: str = ""):
        """
        Load ORB images from the specified directory.
        Removes zero pixels and resizes images to 200x200.
            mst_idir: String, directory containing ORB images.
        Returns a list of MST samples as numpy arrays.
        """
        msts_idir = msts_idir if msts_idir == "" else self.msts_idir
        mst_fns = sorted(glob.glob(os.path.join(self.msts_idir, "*.png")))
        msts = []
        for ofn in mst_fns:
            mst = skimage.io.imread(ofn)[:, :, :3]
            mst = skimage.transform.resize(mst, output_shape=(200, 200, 3))
            mst_nz = mst[mst.sum(axis=2) != 0, :]
            mst_nz *= 255
            msts.append(mst_nz)
        self.msts = msts
        return msts

    def get_pdf(self, samp: np.array,
                ncomp: int = 1, cov_type: str = "full"):
        """
        Estimate the PDF of the sample using a Gaussian Mixture Model.
            samp: Numpy array, sample data, size is nsamples x features.
            ncomp: Integer, number of components in the GMM.
            cov_type: String, type of covariance matrix to use.
        Returns a fitted Gaussian Mixture Model or None if fitting fails.
        """
        # VB-GMM fit
        print("processing pdf")
        samp = samp[samp.sum(axis=1) != 0, :]
        print("DEBUG get_pdf samp.shape:", samp.shape)

        pdf = sklearn.mixture.BayesianGaussianMixture(
            n_components=ncomp, covariance_type=cov_type, max_iter=200,
            random_state=0)
        try:
            pdf.fit(samp)
        except:
            assert False, "Failed to fit GMM"
        return pdf

    def get_pdf_vals(self, pdf: sklearn.mixture.BayesianGaussianMixture,
                     start: int, stop: int, step: int):
        """
        Get PDF values for a range of points in a 3D space.
            pdf: Fitted Gaussian Mixture Model or None for uniform distribution.
            start: Integer, start of the range, included.
            stop: Integer, end of the range, included.
            step: Integer, number of points to generate in each dimension.
        Returns a tuple of points and their corresponding PDF values.
        """
        # Make mesh grids
        crange = np.linspace(start=start, stop=stop, num=step)
        x, y, z = np.meshgrid(crange, crange, crange)
        pts = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
        # Get log scores
        if pdf is None:
            # uniform distribution
            pdf_vals = np.ones(pts.shape[0])
        else:
            ss = pdf.score_samples(pts)
            # Get PDF values
            pdf_vals = np.exp(ss)
        pdf_vals /= (pdf_vals.sum() + EPS)
        return pts, pdf_vals

    def get_kl_distances(self, msts_pdfs_vals: list,
                         img_pdf_vals: np.array):
        """
        Compute KL distances between ORB PDFs and image PDF values.
            orbs_pdfs_vals: List of numpy arrays, PDF values for ORBs.
            img_pdf_vals: Numpy array, PDF values for the image.
        Returns a list of KL distances.
        """
        kld = lambda p, q: np.sum(p * np.log((p + EPS) / (q + EPS)))
        klds = []
        try:
            klds = [kld(p, img_pdf_vals) for p in msts_pdfs_vals]
        except:
            klds = list(range(len(msts_pdfs_vals)))
        return klds

    def get_l1_distances(self, msts_pdfs_vals: list,
                         img_pdf_vals: np.array):
        """
        Compute L1 distances between ORB PDFs and image PDF values.
            orbs_pdfs_vals: List of numpy arrays, PDF values for ORBs.
            img_pdf_vals: Numpy array, PDF values for the image.
        Returns a list of L1 distances.
        """
        ### TODO: we should convert it to integration rather than only values
        # L1 Similarity:
        # We changed l1 as the following:
        # sum(min(f(x) - g(x))) = 0.5 * (sum(abs(f(x))) + sum(abs(g(x))) -
        #                                sum(abs(f(x) - g(x))))
        # If f(x) and g(x) are PDFs:
        # sum(min(f(x) - g(x))) = 0.5 * (1 + 1 - sum(abs(f(x) - g(x)))) =
        #                       = 1 - 0.5 * sum(abs(f(x) - g(x)))
        #
        # L1 distance: sum(abs(fx - gx))
        # Total Variation: 0.5 * sum(abs(fx - gx))
        #
        l1d = [0.5 * np.abs(p - img_pdf_vals).sum() for p in msts_pdfs_vals]
        return l1d

    def get_membership_score(self, dist: np.array):
        """
        Compute membership scores from distances.
            dist: Numpy array, distances.
        Returns a numpy array of membership probabilities.
        """
        dist = np.array(dist)
        memberships = 1 - dist / dist.max()
        mem_prob = sp.special.softmax(memberships)
        return mem_prob
