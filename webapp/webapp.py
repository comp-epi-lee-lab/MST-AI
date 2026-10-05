import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import mst_ai
import skimage
import glob, os
import pickle

MODEL_IDIR = "./model/trial_0000.pckl"
MSTS_IDIR = "/labs_leelab/members/stheanraj/mst/mst_orbs"
CACHE_PATH = "/labs_leelab/members/stheanraj/mst/msts_pdfs_vals.pkl"

st.set_page_config(
    page_title="MST Skin color Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MST Skin Color Analysis")
st.markdown("---")

# MST Scale at the top
st.write(
    "MST scale offers a more holistic view that can be applied across diverse demographics, providing insights into skin sensitivity and adaptability")
st.image('MST_SKIN_TONE_SCALE.png', caption="Monk skin tone (MST) scale", use_column_width=True)
st.markdown("---")

# Create columns
left_column, right_column = st.columns([1, 2])

with left_column:
    # Gray background for input section
    st.markdown("""
        <style>
        .input-section {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.header("Input Options")

        # Upload image only (removed camera input)
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Upload your image", type=['jpg', 'png', 'jpeg'])

        # Use uploaded image
        image = uploaded_image

        # Show uploaded image immediately
        if image is not None:
            st.success("Image loaded successfully!")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Upload an image to get started")

        st.markdown("---")

        # Radio button for lesion selection with default to "No Lesion"
        st.subheader("🔬 Lesion Detection")
        has_lesion = st.radio(
            "Does the image have a lesion?",
            options=[False, True],
            format_func=lambda x: "No Lesion" if not x else "Has Lesion",
            index=0,  # Default to "No Lesion" (index 0)
            horizontal=True
        )

        st.markdown("---")

        # Run button without emoji
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

with right_column:
    st.header("Analysis Results")

    if image is not None and run_analysis:
        with st.spinner("Processing image..."):
            try:
                # Read and process image
                img = skimage.io.imread(image)[:, :, :3]
                org_img = img[:256, :256, :].copy()

                # Initialize MST AI
                mstai = mst_ai.MSTAI(msts_idir=MSTS_IDIR)

                # Get lesion if specified
                lesion = mstai.get_lesion(org_img) if has_lesion else None

                # Get frame
                frame = mstai.get_frame(org_img)

                # Get skin
                skin = mstai.get_skin(img=org_img, lesion=lesion, frame=frame)

                # Get inliers
                inlier = mstai.get_inliers(skin)

                st.success("Image processed successfully!")

                # Load or compute MST PDFs
                mst_fns = sorted(glob.glob(os.path.join(MSTS_IDIR, "*.png")))

                if os.path.exists(CACHE_PATH):
                    with open(CACHE_PATH, "rb") as f:
                        msts_pdfs_vals = pickle.load(f)
                    print("Loaded cached MST PDFs")
                else:
                    # Compute once if cache not found
                    msts = mstai.get_monk_pixels(msts_idir=MSTS_IDIR)
                    msts_pdfs = [mstai.get_pdf(op, ncomp=8) for op in msts]
                    msts_pdfs_vals = [
                        mstai.get_pdf_vals(pdf, start=0, stop=255, step=100)[1]
                        for pdf in msts_pdfs
                    ]
                    with open(CACHE_PATH, "wb") as f:
                        pickle.dump(msts_pdfs_vals, f)
                    print("Computed and saved MST PDFs")

                # Get PDF values for the image
                img_pdf = mstai.get_pdf(inlier.reshape((-1, 3)), ncomp=8)
                _, img_pdf_vals = mstai.get_pdf_vals(img_pdf, start=0, stop=255, step=100)

                # Compute KL distances
                klds = mstai.get_kl_distances(msts_pdfs_vals, img_pdf_vals)

                # Get membership scores with KLD
                memberships_kl = mstai.get_membership_score(klds)
                print("Memberships with KLD:", np.round(memberships_kl, 3))

                # Compute L1 distances
                l1d = mstai.get_l1_distances(msts_pdfs_vals, img_pdf_vals)

                # Get membership scores with L1
                memberships = mstai.get_membership_score(l1d)
                print("Memberships with L1:", np.round(memberships, 3))

                # Sort and get top 3 matches
                sorted_idx = np.argsort(memberships_kl)[::-1][:3]  # largest first
                sorted_imgs = [mst_fns[i] for i in sorted_idx]
                sorted_scores = [memberships_kl[i] for i in sorted_idx]

                st.markdown("---")
                st.markdown("Top Matching MST Tones")
                st.markdown("---")

                # Display results as Rank 1, Rank 2, Rank 3
                for rank, (img_path, score) in enumerate(zip(sorted_imgs, sorted_scores), start=1):
                    col1, col2, col3 = st.columns([1, 2, 3])

                    with col1:
                        st.markdown(f"#### **Rank {rank}**")

                    with col2:
                        st.image(Image.open(img_path), width=150)

                    with col3:
                        st.markdown(f"**Match Score:** `{score:.4f}`")

                        # Extract tone number from filename
                        tone_number = os.path.basename(img_path).split('.')[0].split('_')[-1]
                        st.markdown(f"**MST Tone:** `{tone_number}`")

                        # Progress bar for visual score representation
                        st.progress(float(score))

                    st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)

    elif image is None:
        st.info(" Please upload an image from the left panel to begin analysis")

    else:
        st.info(" Click 'Run Analysis' to process the image")