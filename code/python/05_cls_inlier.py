"""
Current Step:
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
import skimage
import sklearn, sklearn.mixture, sklearn.ensemble, sklearn.exceptions
import torch, torchvision
import multiprocessing as mps

# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
NPS = 38
ISIZE = 256
MODEL_IDIR = "../../model/trial_0000.pckl"
RGB_MST_CIEDE2000_IFNAME = "../../data/rgb_mst_ciede2000.pckl"
CLASSIFICATION_IFNAME = \
    "/mnt/isilon/air/data/melanoma/list/classification.xlsx"
CLASSIFICATION_MEMBERSHIP_ODIR = "../../data/classification_membership_probs/"
CLASSIFICATION_OFNAME = "../../data/classification_membership_probs.xlsx"
INLIER_ODIR = "../../output/classification_inlier/"
CIEDE_NORMALIZER = 96.0
TAU = 0.0183
# Use RGB_MST_CIEDE as a global variable
with open(RGB_MST_CIEDE2000_IFNAME, mode='rb') as f:
    RGB_MST_CIEDE = pickle.load(f)

def lookup_rgb_distances(
    inlier: np.ndarray,
    rgb_mst_ciede: np.ndarray = RGB_MST_CIEDE) -> np.ndarray:
    """
    Retrieve precomputed distances for specific RGB colors.

    Parameters
    ----------
    fname : str
        Filename of the image to extract inlier colors from.
    rgb_mst_ciede : (16777216, 16) ndarray
        Precomputed CIEDE2000 distances between all RGB colors and MSTs.
    Returns
    -------
    distances : (N, 16) ndarray
        Distance rows pulled from `rgb_dists` for each color.
    """
    rgb_mst_ciede = rgb_mst_ciede[:, 6:]
    # Allow float colors scaled 0–1
    r, g, b = tuple(inlier[:, c].astype(np.int64) for c in range(3))
    indices = (r << 16) | (g << 8) | b
    return rgb_mst_ciede[indices]

def mu_sig(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation along axis 0."""
    mu = np.mean(data, axis=0)
    sig = np.std(data, axis=0)
    return mu, sig

def expected_abs(mu, sigma):
    """
    Compute E[|X|] where X ~ N(mu, sigma^2)
    """
    a = mu / (sigma + EPS)
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


class InlierCombiner:
    def __init__(self, org_img_fn: str = ""):
        self.fullpath = org_img_fn
        self.fname = org_img_fn.replace(
            "/mnt/isilon/air/data/melanoma/", "").replace("/", "_")
        # Read the image
        self.org_img = self.read_image(org_img_fn)
        img = self.remove_small_borders(self.org_img, ratio=0.025)
        self.img = np.uint8(skimage.transform.resize(
            img, (ISIZE, ISIZE), anti_aliasing=True) * 255)
        self.transforms = self.make_transforms(isize=ISIZE)
        self.model = self.load_model(MODEL_IDIR)
        self.lesion = self.get_lesion(self.img)
        # if "0000002.jpg" in org_img_fn:
        #     br()
        self.frame = self.get_frame(self.img, border_threshold=0.03)
        self.skin = self.get_skin(self.img, lesion=self.lesion,
                                  frame=self.frame)
        self.inlier_image = self.get_inlier_image(self.skin)
        self.inlier = self.inlier_image[
            self.inlier_image.sum(axis=2) != 0, :]
        self.inlier_ratio = len(self.inlier) / ISIZE / ISIZE
        self.combine = np.concatenate(
            (self.img,
             255 * self.lesion[:, :, np.newaxis].repeat(3, axis=2),
             255 * self.frame[:, :, np.newaxis].repeat(3, axis=2),
             self.skin,
             self.inlier_image), axis=1)
        self.mus, self.sigs = self.get_musig()
        self.membership = self.get_membership_score()

    def read_image(self, img_fn: str):
        """
        Read an image from a file.
            img_fn: String, path to the image file.
        Returns a numpy array of the image.
        """
        img = skimage.io.imread(img_fn)
        return img
    
    def remove_small_borders(self, img: np.array, ratio: float = 0.025):
        """
        Remove small borders from the image.
            img: Numpy array, input image.
            ratio: Float, ratio of the border to be removed.
        Returns a numpy array of the cropped image.
        """
        h, w = img.shape[:2]
        dh = int(h * ratio)
        dw = int(w * ratio)
        cropped_img = img[dh:h-dh, dw:w-dw, :]
        return cropped_img
    
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
    
    def get_frame(self, img: np.array, border_threshold: float):
        """
        Extract frame pixels from the image.
        This function is the same as get_convex_hull (3.5) in exp_005.
            img: Numpy array, input image.
        Returns a binary numpy array (binary mask) of frame pixels.
        """
        gray = skimage.color.rgb2gray(img)
        scale = lambda x: (x - x.min()) / (x.max() - x.min() + EPS)
        hc = scale(gray)
        hc = 0.5 * (1 + np.tanh(4*hc - 2))
        # def detect_convex_hull(mask: np.array):
        #     # Label connected components
        #     lbl = skimage.measure.label(mask, connectivity=2, background=0)
        #     # Accumulate convex hulls per component
        #     chull = np.zeros_like(mask, dtype=bool)
        #     for c in range(1, lbl.max() + 1):
        #         comp = (lbl == c)
        #         if comp.sum() < 1000:
        #             continue
        #         ch = skimage.morphology.convex_hull_image(comp)
        #         if (ch.sum() == (mask.shape[0] * mask.shape[1])):
        #             continue
        #         chull |= ch
        #     return 1.0 * chull
        # chull = detect_convex_hull(bin)
        bin = 1.0 * (hc > border_threshold)
        chull = skimage.morphology.convex_hull_image(bin)
        # If there is no convex hull, maybe the border is white, so the binary
        # image should be inverted, but with half of the threshold
        if np.all(chull == 1):
            bin = 1.0 * (hc < (1-border_threshold/2))
            # chull = detect_convex_hull(bin)
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
    
    def get_inlier_image(self, skin: np.array):
        """
        Extract inlier pixels from the image.
            img: Numpy array, input image, removed lesion and frame.
        Returns a numpy array of inlier pixels.
        """
        inlier_image = skin.copy()
        y = skin[skin.sum(axis=2) != 0, :]
        if y.shape[0] != 0:
            # model = sklearn.linear_model.SGDOneClassSVM(nu=0.15)
            model = sklearn.ensemble.IsolationForest(n_estimators=50)
            # model = sklearn.neighbors.LocalOutlierFactor(n_neighbors=20)
            # lbs = model.fit_predict(skin.reshape((-1, 3)))
            model.fit(y)
            lbs = model.predict(skin.reshape((-1, 3)))
            # print(np.unique(lbs), (lbs == -1).sum(), (lbs == 1).sum())
            # Reconstruct the image with black color for outliers
            inlier_image = np.copy(skin)
            inlier_image[lbs.reshape(skin.shape[:2]) == -1, :] = \
                np.array([0, 0, 0])
        inlier_image = inlier_image.astype(np.uint8)
        # skimage.io.imsave('inlier.png', inlier)
        return inlier_image
    
    def get_musig(self):
        """
        Compute mean and stddev of distances from inlier pixels to MSTs.
            dist: Numpy array, distances.
        Returns a tuple of numpy arrays (mu, sigma).
        """
        mu, sig = np.zeros((10,), dtype=np.float64), np.zeros((10,), dtype=np.float64)
        inlier = self.inlier
        dists = lookup_rgb_distances(inlier)
        dists /= CIEDE_NORMALIZER
        if dists.shape[0] > 0:
            mu, sig = mu_sig(dists)
        return mu, sig
    
    def get_membership_score(self):
        """
        Compute membership scores from distances.
            dist: Numpy array, distances.
        Returns a numpy array of membership probabilities.
        """
        memprob = np.ones((10,), dtype=np.float64) / 10.0
        memprob = membership_probs(self.mus, self.sigs, tau=TAU)
        return memprob

def _mp(fn: str) -> bool:
    combiner = InlierCombiner(org_img_fn=fn)
    out_fn = os.path.join(INLIER_ODIR, 
                          f'{combiner.fname.replace(".jpg", "")}.png')
    skimage.io.imsave(out_fn, combiner.combine.astype(np.uint8))
    result = (out_fn,
              combiner.inlier_ratio,
              combiner.mus,
              combiner.sigs, 
              combiner.membership)
    # Save the results in a separate csv file
    memprob_fn = os.path.join(CLASSIFICATION_MEMBERSHIP_ODIR,
                              f'{combiner.fname.replace(".jpg", "")}.csv')
    df = pandas.DataFrame(columns=['combined', 'inlier_ratio'] +
                          [f'mst_{k}_mu' for k in range(10)] +
                          [f'mst_{k}_sig' for k in range(10)] +
                          [f'mst_{k}_prob' for k in range(10)])
    df.loc[0, 'combined'] = out_fn
    df.loc[0, 'inlier_ratio'] = combiner.inlier_ratio
    for k in range(10):
        df.loc[0, f'mst_{k}_mu'] = combiner.mus[k]
    for k in range(10):
        df.loc[0, f'mst_{k}_sig'] = combiner.sigs[k]
    for k in range(10):
        df.loc[0, f'mst_{k}_prob'] = combiner.membership[k]
    df.to_csv(memprob_fn, index=False)
    return result

def main(argv):
    df = pandas.read_excel(CLASSIFICATION_IFNAME)
    fnames = df['image'].values.tolist()
    rem_fnames = fnames[14000:] # We do not want to start from the beginning
    # results = []
    if NPS == 1:
        for i, fname in enumerate(rem_fnames):
            print(f'{i+1}/{len(fnames)}: {fname}')
            _ = _mp(fname)
            # results.append(res)
    else:
        with mps.Pool(NPS) as pool:
            _ = pool.map(_mp, rem_fnames)
    # Restore results from saved CSV files
    new_cols = ['combined', 'inlier_ratio'] + \
        [f'mst_{k}_mu' for k in range(10)] + \
        [f'mst_{k}_sig' for k in range(10)] + \
        [f'mst_{k}_prob' for k in range(10)]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None
    for fname in fnames:
        fn = fname.replace("/mnt/isilon/air/data/melanoma/", "")
        fn = fn.replace("/", "_").replace(".jpg", "") + ".csv"
        csv_fname = os.path.join(CLASSIFICATION_MEMBERSHIP_ODIR, fn)
        if not os.path.exists(csv_fname):
            continue
        # Update the df with the results
        df_tmp = pandas.read_csv(csv_fname)
        df.loc[df['image'] == fname, new_cols] = df_tmp.iloc[0].values
        
    df.to_excel(CLASSIFICATION_OFNAME, index=False)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')