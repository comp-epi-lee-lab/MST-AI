import  pickle
import os,glob
import mst_ai

MSTS_IDIR = "/labs_leelab/members/stheanraj/mst/mst_orbs"
CACHE_PATH = "/labs_leelab/members/stheanraj/mst/msts_pdfs_vals.pkl"

mstai = mst_ai.MSTAI(msts_idir=MSTS_IDIR)

msts = mstai.get_monk_pixels(msts_idir=MSTS_IDIR)
msts_pdfs = [mstai.get_pdf(op, ncomp=8) for op in msts]
msts_pdfs_vals = [
    mstai.get_pdf_vals(pdf, start=0, stop=255, step=100)[1]
    for pdf in msts_pdfs
]

with open(CACHE_PATH, "wb") as f:
    pickle.dump(msts_pdfs_vals, f)
