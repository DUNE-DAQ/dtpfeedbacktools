#!/usr/bin/env python

from dtpfeedbacktools.rawdatamanager import RawDataManager
import os.path
import sys
import rich
from rich.table import Table
import logging
import click
from rich import print
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
plt.style.use('ggplot')

header_labels = {'offline_ch':("offline channel", ""), 'crate_no':("crate", ""), 'slot_no':("slot", ""), 'fiber_no':("fiber", ""), 'wire_no':("wire index", ""), 'ts':("time stamp", " [tts]")}
hit_labels = {'start_time':("start time", " [ticks]"), 'end_time':("end time", " [ticks]"), 'peak_time':("peak time", " [ticks]"), 'peak_adc':("peak adc", " [ADCs]"), 'hit_continue':("hit continue", ""), 'sum_adc':("sum adc", " [adc]")}
pedinfo_labels = {'median':("median", " [ADCs]"), 'accumulator':("accumulator", "")}
all_plots = {"header":header_labels, "hit":hit_labels, "pedinfo":pedinfo_labels}

NS_PER_TICK = 16

def hist_plot(rtp_df, var, axs, label_dict, nbins=None, ylabel="# hits"):
    xmin = np.min(rtp_df[var])-.5
    xmax = np.max(rtp_df[var])+.5
    if nbins == None:
        nbins = int(xmax-xmin+1)

    bins = np.linspace(xmin, xmax, nbins)
    axs.hist(rtp_df[var], histtype='barstacked', color="dodgerblue", alpha=0.7, bins=bins)
    #axs.set_title(label_dict[var][0])
    axs.set_yscale("log")
    axs.set_xlim(xmin-1, xmax+1)
    axs.set_xlabel(label_dict[var][0]+label_dict[var][1], loc="right")
    axs.set_ylabel(ylabel, loc="top")

def density_scatter(rtp_df, varx, vary, axs, label_dictx, label_dicty, nbinsx=None, nbinsy=None, xlabel="wire index", ylabel="", clabel="# hits"):
    xmin = np.min(rtp_df[varx])-.5
    xmax = np.max(rtp_df[varx])+.5
    if nbinsx == None:
        nbinsx = int(xmax-xmin+1)
    binsx = np.linspace(xmin, xmax, nbinsx)

    ymin = np.min(rtp_df[vary])-.5
    ymax = np.max(rtp_df[vary])+.5
    if nbinsy == None:
        nbinsy = int(ymax-ymin+1)
    binsy = np.linspace(ymin, ymax, nbinsy)

    hist2d, xedges, yedges = np.histogram2d(rtp_df[varx], rtp_df[vary], bins=[binsx, binsy])
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2 
    
    heat_map = np.array([[x, y, hist2d[i,j]] for i, x in enumerate(xcenters) for j, y in enumerate(ycenters) if hist2d[i,j] != 0])

    im=axs.scatter(heat_map[:,0], heat_map[:,1], c=heat_map[:,2], cmap="jet", norm=matplotlib.colors.LogNorm(), alpha=0.6, s=5)
    axs.set_xlim(xmin-.5, xmax+.5)
    axs.set_ylim(ymin-.5, ymax+.5)
    axs.set_xlabel(label_dictx[varx][0]+label_dictx[varx][1], loc="right")
    axs.set_ylabel(label_dicty[vary][0]+label_dicty[vary][1], loc="top")
    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label(clabel)

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return np.array([overlap_true, overlap_time])

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('-s', '--save_df', is_flag=True, default=False)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('-n', '--n_lines', help="Number of lines to be ploted, -1 for full file", default=-1)
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default=None)
@click.option('-o', '--outpath', default="./validation/")

def cli(interactive: bool, save_df: bool, n_lines: int, files_path: str, map_id: str, outpath: str) -> None:

    rdm = RawDataManager(files_path, ch_map_id=map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    if (n_lines == -1):
        n_blocks = -1
    else:
        n_blocks = n_lines//3

    rich.print(tp_files)

    for i, f in enumerate(tp_files):
        rtp_df = rdm.load_tps(f, n_blocks, 0)
        rich.print(rtp_df)
        if save_df:
            rtp_df.to_hdf("rtp_"+f.replace(".out", "")+".hdf5", key="rtp")
        
        #outname = os.path.join(outpath, f.replace(".out", ".pdf"))
        #pdf = matplotlib.backends.backend_pdf.PdfPages(outname)

        fig = plt.figure(figsize=(30,14))
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.5, height_ratios=[1.5,1.5,1.5])
        axs = gs.subplots()

        for j, var in enumerate(hit_labels.keys()):
            if((var == "peak_adc")or(var == "sum_adc")):
                nbins = 100
                nbinsy = 100
            else:
                nbins = None
                nbinsy = None
            hist_plot(rtp_df, var, axs[0,j], hit_labels, nbins)
            density_scatter(rtp_df, "wire_no", var, axs[1,j], header_labels, hit_labels, nbinsy)
            density_scatter(rtp_df, "offline_ch", var, axs[2,j], header_labels, hit_labels, nbinsy)

        #pdf.savefig()
        plt.savefig(os.path.join(outpath, f.replace(".out", "_hits.png")), dpi=500)
        plt.close()

        fig = plt.figure(figsize=(14,8))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)
        axs = gs.subplots()

        for j, var in enumerate(header_labels.keys()):
            if(var == "ts"):
                nbins = 100
            else:
                nbins = None
            hist_plot(rtp_df, var, axs[int(j>2),j-3*int(j>2)], header_labels, nbins, ylabel="# hit packets")

        #pdf.savefig()
        plt.savefig(os.path.join(outpath, f.replace(".out", "_header.png")), dpi=500)
        plt.close()

        fig = plt.figure(figsize=(9.5,3.5))
        gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.4)
        axs = gs.subplots()

        hist_plot(rtp_df, "median", axs[0], pedinfo_labels, nbins=100, ylabel="# hit packets")
        hist_plot(rtp_df, "accumulator", axs[1], pedinfo_labels, ylabel="# hit packets")

        #pdf.savefig()
        plt.savefig(os.path.join(outpath, f.replace(".out", "_ped.png")), dpi=500)
        plt.close()

        #pdf.close()

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
