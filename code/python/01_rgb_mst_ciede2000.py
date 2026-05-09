#!python

"""
Current Step:
    Computing CIEDE2000 distances between all RGB pixels and all MSTs.
    Previous plan: Removing gradients from Monk ORBs. But we found it really
    inefficient since we have the swatches. All the orbs are made by the 
    CIELAB color space. Hence, this space is perceptually closer to human
    interpretation of the skin colors.
"""

import sys, os
import pickle
import time
import numpy as np

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
RGB_MST_CIEDE2000_OFNAME = "../../data/rgb_mst_ciede2000.pckl"

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert gamma-corrected sRGB values to linear light."""
    rgb = np.asarray(rgb, dtype=np.float64)
    threshold = 0.04045
    below = rgb <= threshold
    above = ~below
    out = np.empty_like(rgb, dtype=np.float64)
    out[below] = rgb[below] / 12.92
    out[above] = ((rgb[above] + 0.055) / 1.055) ** 2.4
    return out


def _f_xyz(t: np.ndarray) -> np.ndarray:
    """Helper non-linearity used in the CIELAB conversion."""
    delta = 6.0 / 29.0
    out = np.empty_like(t)
    higher = t > delta ** 3
    out[higher] = np.cbrt(t[higher])
    out[~higher] = t[~higher] / (3 * delta ** 2) + 4 / 29
    return out

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an array of sRGB colors (0-255) to CIELAB."""
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = rgb / 255.0
    linear = _srgb_to_linear(rgb)
    xyz = linear @ SRGB_TO_XYZ.T
    xyz = xyz / XYZ_WHITEPOINT
    f = _f_xyz(xyz)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack((L, a, b), axis=1)

def ciede2000(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute the CIEDE2000 distance from lab values to a reference."""
    lab = np.asarray(lab, dtype=np.float64)
    L1, a1, b1 = lab[:, 0], lab[:, 1], lab[:, 2]
    L2, a2, b2 = ref

    kL = kC = kH = 1.0

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)

    C7 = C_bar ** 7
    G = 0.5 * (1 - np.sqrt(C7 / (C7 + 25 ** 7)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = np.hypot(a1_prime, b1)
    C2_prime = np.hypot(a2_prime, b2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    h_diff = h2_prime - h1_prime
    h_sum = h1_prime + h2_prime

    delta_h_prime = np.where(
        np.abs(h_diff) <= 180,
        h_diff,
        h_diff - np.sign(h_diff) * 360,
    )
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(
        np.radians(delta_h_prime / 2)
    )

    L_bar_prime = 0.5 * (L1 + L2)
    H_bar_prime = np.where(
        (np.abs(h_diff) > 180) & (h_sum < 360),
        (h_sum + 360) / 2,
        np.where((np.abs(h_diff) > 180) & (h_sum >= 360), (h_sum - 360) / 2, h_sum / 2),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(H_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * H_bar_prime))
        + 0.32 * np.cos(np.radians(3 * H_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * H_bar_prime - 63))
    )

    delta_theta = 30 * np.exp(-(((H_bar_prime - 275) / 25) ** 2))
    R_C = 2 * np.sqrt((C_bar_prime ** 7) / (C_bar_prime ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * ((L_bar_prime - 50) ** 2)) / np.sqrt(20 + (L_bar_prime - 50) ** 2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    delta_E = np.sqrt(
        (delta_L_prime / (kL * S_L)) ** 2
        + (delta_C_prime / (kC * S_C)) ** 2
        + (delta_H_prime / (kH * S_H)) ** 2
        + R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )
    return delta_E.astype(np.float32)

def compute_mst_distance_table(mst_dir: str) -> np.ndarray:
    """Return an array of RGB, Lab, and CIEDE2000 distances to each MST."""
    mst_labs = rgb_to_lab(MST_RGB_255)

    values = np.arange(256, dtype=np.uint16)
    r, g, b = np.meshgrid(values, values, values, indexing="ij")
    rgb = np.stack((r, g, b), axis=-1).reshape(-1, 3).astype(np.float32)

    lab = rgb_to_lab(rgb)

    result = np.empty((rgb.shape[0], 16), dtype=np.float32)
    result[:, 0:3] = rgb
    result[:, 3:6] = lab.astype(np.float32)

    for idx, mst_lab in enumerate(mst_labs):
        result[:, 6 + idx] = ciede2000(lab, mst_lab)

    return result

def main(argv) -> None:
    data = compute_mst_distance_table("./mst_orbs")
    print(data.shape)
    with open(RGB_MST_CIEDE2000_OFNAME, mode='wb') as f:
        pickle.dump(data, f)
    return None

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
