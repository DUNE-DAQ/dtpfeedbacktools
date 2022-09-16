#!/usr/bin/env python

from dtpfeedbacktools.rawdatamanager import RawDataManager
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

header = ['offline_ch', 'crate_no', 'slot_no', 'fiber_no', 'wire_no', 'flags']
hit = ['start_time', 'end_time', 'peak_time', 'peak_adc', 'hit_continue', 'tp_flags', 'sum_adc']
pedinfo = ['median', 'accumulator']
all_plots = {"header":header, "hit":hit, "pedinfo":pedinfo}

NS_PER_TICK = 16

def validation_hist(rtp_df, var):
    xmin = np.min(rtp_df[var])-.5
    xmax = np.max(rtp_df[var])+.5
    fig = plt.figure()
    plt.hist(rtp_df[var], histtype='step', bins=np.linspace(xmin, xmax, int(xmax-xmin+1)))
    plt.title(var)
    plt.yscale("log")
    plt.xlim(xmin-1, xmax+1)
    plt.xlabel(var)
    plt.ylabel("# hit packets")
    plt.grid()
    return fig

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return np.array([overlap_true, overlap_time])

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
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

def cli(interactive: bool, n_lines: int, files_path: str, map_id: str, outpath: str) -> None:

    rdm = RawDataManager(files_path, ch_map_id=map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    if (n_lines == -1):
        n_blocks = -1
    else:
        n_blocks = n_lines//3

    rich.print(tp_files)
    
    for i, f in enumerate(tp_files):
        rtp_df = rdm.load_tps(f, n_blocks, 0)
        for set in all_plots:
            outname = outpath+f.replace(".out", "_"+set+".pdf")
            pdf = matplotlib.backends.backend_pdf.PdfPages(outname)
            for var in all_plots[set]:
                fig = validation_hist(rtp_df, var)
                pdf.savefig(fig)
                plt.close()
            pdf.close()

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
