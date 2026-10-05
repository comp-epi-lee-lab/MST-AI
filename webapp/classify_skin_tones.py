"""
MST Skin Tone Batch Classifier
Classifies a dataset of images by Monk Skin Tone (MST) and organizes
them into subfolders based on the top-1 KL membership score.

Usage:
    python classify_skin_tones.py \
        --input_dir  /path/to/images \
        --output_dir /path/to/sorted_output \
        --msts_idir  /path/to/mst_orbs \
        --cache_path /path/to/msts_pdfs_vals.pkl   # optional, speeds up reruns
        --has_lesion                                # optional flag
        --ext jpg png jpeg                          # optional, default: jpg png jpeg
"""

import argparse
import glob
import logging
import os
import pickle
import shutil

import numpy as np
import skimage.io
from tqdm import tqdm

import mst_ai

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_or_compute_mst_pdfs(mstai: mst_ai.MSTAI,
                              mst_fns: list,
                              cache_path: str) -> list:
    """Return cached MST PDF values, computing & caching them if needed."""
    if cache_path and os.path.exists(cache_path):
        log.info("Loading cached MST PDFs from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log.info("Computing MST PDFs (one-time cost) …")
    msts = mstai.get_monk_pixels(msts_idir=mstai.msts_idir)
    msts_pdfs = [mstai.get_pdf(op, ncomp=8) for op in msts]
    msts_pdfs_vals = [
        mstai.get_pdf_vals(pdf, start=0, stop=255, step=100)[1]
        for pdf in msts_pdfs
    ]

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(msts_pdfs_vals, f)
        log.info("Saved MST PDF cache to %s", cache_path)

    return msts_pdfs_vals


def classify_image(img_path: str,
                   mstai: mst_ai.MSTAI,
                   mst_fns: list,
                   msts_pdfs_vals: list,
                   has_lesion: bool = False) -> tuple[str, float]:
    """
    Classify a single image.

    Returns
    -------
    (tone_label, top1_score)
        tone_label : str  – e.g. "MST_01"
        top1_score : float
    """
    img = skimage.io.imread(img_path)[:, :, :3]
    org_img = img[:256, :256, :].copy()

    lesion = mstai.get_lesion(org_img) if has_lesion else None
    frame  = mstai.get_frame(org_img)
    skin   = mstai.get_skin(img=org_img, lesion=lesion, frame=frame)
    inlier = mstai.get_inliers(skin)

    img_pdf = mstai.get_pdf(inlier.reshape((-1, 3)), ncomp=8)
    _, img_pdf_vals = mstai.get_pdf_vals(img_pdf, start=0, stop=255, step=100)

    klds        = mstai.get_kl_distances(msts_pdfs_vals, img_pdf_vals)
    memberships = mstai.get_membership_score(klds)        # softmax, higher = better match

    top1_idx    = int(np.argmax(memberships))
    top1_score  = float(memberships[top1_idx])

    # Derive a clean label from the ORB filename, e.g. "mst_01" → "MST_01"
    tone_label = (
        os.path.basename(mst_fns[top1_idx])   # e.g. "orb_01.png"
        .split(".")[0]                          # "orb_01"
        .replace("orb", "MST")                 # "MST_01"  (adjust if naming differs)
        .upper()
    )

    return tone_label, top1_score


def collect_images(input_dir: str, extensions: list[str]) -> list[str]:
    """Recursively collect all image paths with the given extensions."""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(input_dir, "**", f"*.{ext}"), recursive=True))
    return sorted(set(paths))


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── resolve MST ORB filenames ────────────────────────────────────────────
    mst_fns = sorted(glob.glob(os.path.join(args.msts_idir, "*.png")))
    if not mst_fns:
        raise FileNotFoundError(f"No ORB PNGs found in: {args.msts_idir}")
    log.info("Found %d MST ORB references.", len(mst_fns))

    # ── initialise MSTAI ────────────────────────────────────────────────────
    mstai = mst_ai.MSTAI(msts_idir=args.msts_idir)

    # ── MST PDFs (cached) ───────────────────────────────────────────────────
    msts_pdfs_vals = load_or_compute_mst_pdfs(mstai, mst_fns, args.cache_path)

    # ── collect input images ─────────────────────────────────────────────────
    img_paths = collect_images(args.input_dir, args.ext)
    if not img_paths:
        raise FileNotFoundError(
            f"No images with extensions {args.ext} found in: {args.input_dir}"
        )
    log.info("Found %d images to classify.", len(img_paths))

    # ── pre-create output folders for every known tone ───────────────────────
    for fn in mst_fns:
        tone = (
            os.path.basename(fn).split(".")[0].replace("orb", "MST").upper()
        )
        os.makedirs(os.path.join(args.output_dir, tone), exist_ok=True)

    # ── classify & copy ──────────────────────────────────────────────────────
    results = []          # list of dicts for the summary CSV
    errors  = []

    for img_path in tqdm(img_paths, desc="Classifying", unit="img"):
        try:
            tone_label, score = classify_image(
                img_path, mstai, mst_fns, msts_pdfs_vals, args.has_lesion
            )

            dest_dir  = os.path.join(args.output_dir, tone_label)
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))

            # Avoid silent overwrites when two source images share a filename
            if os.path.exists(dest_path):
                base, ext_ = os.path.splitext(os.path.basename(img_path))
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext_}")
                    counter  += 1

            shutil.copy2(img_path, dest_path)

            results.append({
                "source":     img_path,
                "dest":       dest_path,
                "tone_label": tone_label,
                "kl_score":   round(score, 6),
            })
            log.debug("%-60s  →  %s  (%.4f)", img_path, tone_label, score)

        except Exception as exc:
            log.warning("FAILED %s: %s", img_path, exc)
            errors.append({"source": img_path, "error": str(exc)})

    # ── summary CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "classification_results.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "dest", "tone_label", "kl_score"])
        writer.writeheader()
        writer.writerows(results)

    if errors:
        err_path = os.path.join(args.output_dir, "errors.csv")
        with open(err_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "error"])
            writer.writeheader()
            writer.writerows(errors)
        log.warning("%d image(s) failed — see %s", len(errors), err_path)

    # ── tone distribution summary ────────────────────────────────────────────
    from collections import Counter
    tone_counts = Counter(r["tone_label"] for r in results)
    log.info("─" * 50)
    log.info("Classification complete.")
    log.info("  Total processed : %d", len(results))
    log.info("  Total failed    : %d", len(errors))
    log.info("  Tone distribution:")
    for tone in sorted(tone_counts):
        log.info("    %-12s : %d images", tone, tone_counts[tone])
    log.info("  Results saved to : %s", csv_path)
    log.info("─" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a dataset of images by Monk Skin Tone (MST) "
                    "and copy them into per-tone subfolders."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Root directory containing the images to classify (searched recursively)."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root output directory. Subfolders (MST_01 … MST_10) are created automatically."
    )
    parser.add_argument(
        "--msts_idir", required=True,
        help="Directory containing the Monk ORB reference PNG images."
    )
    parser.add_argument(
        "--cache_path", default=None,
        help="Optional path to a .pkl cache for MST PDFs (saves recomputation)."
    )
    parser.add_argument(
        "--has_lesion", action="store_true", default=False,
        help="Run the lesion-extraction step before skin analysis."
    )
    parser.add_argument(
        "--ext", nargs="+", default=["jpg", "jpeg", "png"],
        help="Image file extensions to include (default: jpg jpeg png)."
    )

    args = parser.parse_args()
    main(args)