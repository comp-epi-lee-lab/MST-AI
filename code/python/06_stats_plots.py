"""
Current Step:
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
import skimage
import plotly, plotly.express as px, plotly.graph_objects as go
import plotly.subplots
# import multiprocessing as mps

# import aux_v05 as aux

# globals
br = breakpoint
EPS = 1e-6
e = lambda: os._exit(0)
NPS = 1
ISIZE = 256
INLIER_THRESH = 0.05
MAX_RANK, PROB_QUANTILE = 3, 0.7
CLASSIFICATION_IFNAME = "../../data/classification_membership_probs.xlsx"
CLASSIFICATION_PCKL = "../../output/classification_membership_probs.pckl"
CLASSIFICATION_OFNAME = "../../data/classification_ranks.xlsx"
INLIER_IDIR = "../../output/classification_inlier/"
PLOTS_ODIR = "../../output/stats_plots/"
os.makedirs(PLOTS_ODIR, exist_ok=True)

def rank_stats(df, max_rank=MAX_RANK):
    """
    This method tries to plot based on the ranks of membership probabilities.
    """
    ## Plot rank-1, rank-2, rank-3 membership probabilities
    mst_cols = [col for col in df.columns if col.endswith('_prob')]
    ranks = df[mst_cols].values.argsort(axis=1)
    fig_abs = plotly.subplots.make_subplots(
        rows=MAX_RANK, cols=1,
        subplot_titles=[f'Rank-{i}' for i in range(1, MAX_RANK + 1)])
    fig_nor = plotly.subplots.make_subplots(
        rows=MAX_RANK, cols=1,
        subplot_titles=[f'Rank-{i}' for i in range(1, MAX_RANK + 1)])
    for rc in range(1, MAX_RANK + 1):
        rank = ranks[:, -rc] + 1.0
        # Absolute values
        fig_abs.add_trace(
            go.Histogram(
                x=rank,
                xbins=dict(start=0.5, end=10.5, size=1), name=f'Rank-{rc}',
                texttemplate="%{y}", textposition="outside"),
            row=rc, col=1)
        fig_abs.update_xaxes(title_text="MST Membership", 
                            tickmode='linear', tick0=1, dtick=1,
                            row=rc, col=1)
        fig_abs.update_yaxes(title_text="Number of Images",
                            range=[0, 32000],
                            row=rc, col=1)
        # Normalized values
        fig_nor.add_trace(
            go.Histogram(
                x=rank,
                xbins=dict(start=0.5, end=10.5, size=1), name=f'Rank-{rc}',
                histnorm="probability",
                texttemplate="%{y:.2f}", textposition="outside"),
            row=rc, col=1)
        fig_nor.update_xaxes(title_text="MST Membership", 
                            tickmode='linear', tick0=1, dtick=1,
                            row=rc, col=1)
        fig_nor.update_yaxes(title_text="Proportion of Images", 
                             range=[0, 0.55],
                            row=rc, col=1)
    fig_abs.write_html(
        os.path.join(PLOTS_ODIR, 'membership_absolute_rank.html'))
    fig_nor.write_html(
        os.path.join(PLOTS_ODIR, 'membership_normalized_rank.html'))
    ## Show how many images are used in each MST training and validation
    rank = (ranks[:, -MAX_RANK:] + 1.0).flatten()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=rank,
            xbins=dict(start=0.5, end=10.5, size=1), 
            name=f'Rank-1 to Rank-{MAX_RANK}',
            texttemplate="%{y}", textposition="outside")
    )
    fig.update_xaxes(title_text="MST Membership", 
                     tickmode='linear', tick0=1, dtick=1)
    fig.update_yaxes(title_text="Number of Images", range=[0, 55000])
    fig.update_layout(
        title=f'Membership Histogram for Top {MAX_RANK} Ranks')
    fig.write_html(
        os.path.join(PLOTS_ODIR, 
                     f'membership_histogram_top_{MAX_RANK}_ranks.html'))
    return True

def elbow_method(arr):
    """
    This method uses the Elbow method to find significant ranks.
    Args:
        arr: (N, M) array of membership probabilities for N samples and M MSTs
    Returns:
        ranks: list of arrays, each array contains the significant ranks
               for the corresponding sample, sorted from highest to lowest.
    """
    arr_sorted = np.sort(arr, axis=1)
    elbow_index = np.argmax(arr_sorted[:, 1:] - arr_sorted[:, :-1], 
                            axis=1)
    elbow = arr_sorted[np.arange(arr_sorted.shape[0]),
                       elbow_index].reshape(-1, 1)
    arr_sorted = np.sort(arr, axis=1)[:, ::-1]
    arr_argsorted = arr.argsort(axis=1)[:, ::-1]
    ranks = [arr_argsorted[c, :][arr_sorted[c, :] > elbow[c]].tolist()
             for c in range(arr.shape[0])]
    return ranks

def prob_stats(df, max_rank=MAX_RANK):
    """
    This method tries to plot based on the percentiles 
    of membership probabilities.
    """
    ## Plot rank-1, rank-2, rank-3 membership probabilities
    mst_cols = [col for col in df.columns if col.endswith('_prob')]
    msts = df[mst_cols].values
    ## Using Elbow method to find the top probabilities.
    # The ranks are sorted from highest to lowest.
    rank_sorted_list = elbow_method(msts)
    lenr = np.array([len(r) for r in rank_sorted_list])
    ## Plotting of the number of MSTs used for each image
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=lenr,
            xbins=dict(start=0.5, end=10.5, size=1), 
            texttemplate="%{y}", textposition="outside")
    )
    fig.update_xaxes(title_text="Number of MSTs used per Image", 
                     tickmode='linear', tick0=1, dtick=1)
    fig.update_yaxes(title_text="Number of Images", range=[0, 55000])
    fig.update_layout(
        title='Histogram of Number of MSTs used per Image')
    fig.write_html(
        os.path.join(PLOTS_ODIR, 
                     f'membership_histogram_number_of_msts_used.html'))
    ## Plot absolute and normalized ranks
    fig_abs = plotly.subplots.make_subplots(
        rows=max_rank, cols=1,
        subplot_titles=[f'Rank-{i}' for i in range(1, max_rank + 1)])
    fig_nor = plotly.subplots.make_subplots(
        rows=max_rank, cols=1,
        subplot_titles=[f'Rank-{i}' for i in range(1, max_rank + 1)])
    for rc in range(max_rank):
        # Rank-(rc+1)
        rank = np.array([r[rc] for r in rank_sorted_list if len(r) > rc]) + 1.0
        # Find upper values
        ymax = np.unique(rank, return_counts=True)[1].max()
        ymax += (0.2 * ymax)
        # Absolute values
        fig_abs.add_trace(
            go.Histogram(
                x=rank,
                xbins=dict(start=0.5, end=10.5, size=1), name=f'Rank-{rc+1}',
                texttemplate="%{y}", textposition="outside"),
            row=rc+1, col=1)
        fig_abs.update_xaxes(title_text="MST Membership", 
                            tickmode='linear', tick0=1, dtick=1,
                            row=rc+1, col=1)
        fig_abs.update_yaxes(title_text="Number of Images",
                            range=[0, ymax],
                            row=rc+1, col=1)
        fig_abs.layout.annotations[rc].text = f"Rank-{rc+1} (n={len(rank)})"
        # Normalized values
        freq = np.unique(rank, return_counts=True)[1]
        rank_prob = freq / freq.sum()
        ymax = rank_prob.max()
        ymax += (0.2 * ymax)
        fig_nor.add_trace(
            go.Histogram(
                x=rank,
                xbins=dict(start=0.5, end=10.5, size=1), name=f'Rank-{rc+1}',
                histnorm="probability",
                texttemplate="%{y:.2f}", textposition="outside"),
            row=rc+1, col=1)
        fig_nor.update_xaxes(title_text="MST Membership", 
                            tickmode='linear', tick0=1, dtick=1,
                            row=rc+1, col=1)
        fig_nor.update_yaxes(title_text="Proportion of Images", 
                            range=[0, ymax],
                            row=rc+1, col=1)
        fig_nor.layout.annotations[rc].text = f"Rank-{rc+1} (n={len(rank)})"
    fig_abs.write_html(
        os.path.join(PLOTS_ODIR, 'membership_absolute_prob.html'))
    fig_nor.write_html(
        os.path.join(PLOTS_ODIR, 'membership_normalized_prob.html'))
    ## Show how many images are used in each MST training and validation
    rank = sum([r[:max_rank] for r in rank_sorted_list], [])
    rank = np.array(rank) + 1.0
    ymax = np.unique(rank, return_counts=True)[1].max()
    ymax += (0.1 * ymax)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=rank,
            xbins=dict(start=0.5, end=10.5, size=1), 
            name=f'Rank-1 to Rank-{max_rank}',
            texttemplate="%{y}", textposition="outside"))
    fig.update_xaxes(title_text="MST Membership", 
                     tickmode='linear', tick0=1, dtick=1)
    fig.update_yaxes(title_text="Number of Images", range=[0, ymax])
    fig.update_layout(
        title=f'Membership Histogram for Top {max_rank} Ranks')
    fig.write_html(
        os.path.join(PLOTS_ODIR, 
                     f'membership_histogram_top_{max_rank}_probs.html'))
    return rank_sorted_list

def plot_ratio(df):
    # Print how many positive and negative samples are in each MST
    print("MST,\tTotal,\tBenign,\tMalignant")
    bar = []
    for c in range(10):
        lbs = df[df[f"mst_{c+1}"] == 1].label
        bar.append([len(lbs), sum(lbs==0), sum(lbs==1)])
        print(f"{c+1},\t{bar[-1][0]},\t{bar[-1][1]},\t{bar[-1][2]}")
    bar = np.array(bar)
    # Stacked Bar Plot
    fig = go.Figure([
        go.Bar(name="Benign", x=np.arange(1, 11), 
               y=100.0 * bar[:, 1] / bar[:, 0]),
        go.Bar(name="Malignant", x=np.arange(1, 11),
               y=100.0 * bar[:, 2] / bar[:, 0])
    ])
    fig.update_layout(
        barmode='stack',
        title='Benign vs Malignant Percentage per MST',
        xaxis_title='MST',
        yaxis_title='Diagnose Percentage'
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.write_html(os.path.join(PLOTS_ODIR, "benign_malignant.html"))
    # Malignancy Ratios Plot
    fig = go.Figure([
        go.Bar(name="Malignant%", 
               x=np.arange(1, 11), y=100.0 * (bar[:, 2] / bar[:, 0]),
               marker_color='red')
    ])
    fig.update_layout(
        title='Malignancy Percentage per MST',
        xaxis_title='MST',
        yaxis_title='Malignancy Percentage'
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.write_html(os.path.join(PLOTS_ODIR, "malignancy_ratio.html"))
    return bar
        

def main(argv):
    if os.path.exists(CLASSIFICATION_PCKL):
        with open(CLASSIFICATION_PCKL, 'rb') as pckl_f:
            df = pickle.load(pckl_f)
    else:
        df = pandas.read_excel(CLASSIFICATION_IFNAME, index_col=None)
        with open(CLASSIFICATION_PCKL, 'wb') as pckl_f:
            pickle.dump(df, pckl_f)
    df = df.drop(columns="index")
    ## Plot inlier ratios histogram in Plotly
    inlier_ratios = df['inlier_ratio'].to_numpy()
    fig = px.histogram(
        inlier_ratios, nbins=20,
        title='Inlier Ratios Histogram',
        labels={'value': 'Inlier Ratio', 'count': 'Number of Images'})
    # Compute the 10% percentile and plot it as a vertical line
    fig.add_vline(x=INLIER_THRESH, line_dash="dash", line_color="red",
                  annotation_text="10th Percentile",
                  annotation_position="top right")
    fig.write_html(os.path.join(PLOTS_ODIR, 'inlier_ratios_histogram.html'))
    # Print the number of samples below and above the 10% percentile
    num_below = np.sum(inlier_ratios < INLIER_THRESH)
    num_above = np.sum(inlier_ratios >= INLIER_THRESH)
    print(f'Number of samples below 10% percentile: {num_below}')
    print(f'Number of samples above 10% percentile: {num_above}')
    ## Plot membership probabilities for each MST
    df_inlier = df[df['inlier_ratio'] >= INLIER_THRESH].copy()
    df_inlier.reset_index(drop=True, inplace=True)
    mst_cols = [col for col in df.columns if col.endswith('_prob')]
    fig = go.Figure()
    for mst_col in mst_cols:
        membership_probs = df_inlier[mst_col].to_numpy()
        fig.add_trace(go.Histogram(
            x=membership_probs,
            name=mst_col,
            opacity=0.75
        ))
        fig.update_layout(
            title='Membership Probabilities Histogram for all MSTs',
            xaxis_title='Membership Probability',
            yaxis_title='Number of Images',
            barmode='overlay'
        )
    fig.write_html(os.path.join(PLOTS_ODIR, 
                               'membership_probs_histogram_all_msts.html'))
    # Plot rank-based statistics
    rank_stats(df_inlier)
    # Plot probability-based statistics
    ranks = prob_stats(df_inlier)
    # Save the ranks into an Excel file
    for c in range(10):
        df_inlier[f'mst_{c+1}'] = 0
    for c in range(len(ranks)):
        df_inlier.loc[c, [f'mst_{i+1}' for i in ranks[c][:MAX_RANK]]] = 1
    df_inlier.to_excel(CLASSIFICATION_OFNAME, index=False)
    plot_ratio(df_inlier)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')