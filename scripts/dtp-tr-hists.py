#!/usr/bin/env python

from dtpemulator.tpgmanager import TPGManager

from dtpfeedbacktools.datamanager import DataManager
import sys
import rich
import logging
import click
from rich import print
from pathlib import Path
from pylab import cm
from matplotlib import colors

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf

fir_shift = 15

hit_labels = {'start_time':("start time", " [ticks]"), 'end_time':("end time", " [ticks]"), 'peak_time':("peak time", " [ticks]"), 'peak_adc':("peak adc", " [ADCs]"), 'hit_continue':("hit continue", ""), 'sum_adc':("sum adc", " [adc]")}

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def rms(array):
    return np.sqrt(np.sum(np.power(array.astype(int), 2))/len(array.astype(int)))


#------------------------------------------------------------------------------
def hist_plot(tp_df, emu_df, var, label_dict, nbins=None, ylabel="# hits", pdf=False):
    xmin = np.min(tp_df[var])-.5
    xmax = np.max(tp_df[var])+.5
    if nbins == None:
        nbins = int(xmax-xmin+1)

    bins = np.linspace(xmin, xmax, nbins)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure()

    plt.hist(tp_df[var], histtype='barstacked', color="dodgerblue", alpha=0.7, bins=bins, label="FWTP")
    plt.hist(emu_df[var], histtype='barstacked', color="red", alpha=0.5, bins=bins, label="Emulation")
    plt.yscale("log")
    plt.xlim(xmin-1, xmax+1)
    plt.xlabel(label_dict[var][0]+label_dict[var][1], loc="right")
    plt.ylabel(ylabel, loc="top")

    legend = plt.legend(fontsize=8, loc="upper right")
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_alpha(0.8)
    frame.set_linewidth(0)

    if pdf:
        pdf.savefig()

    plt.show()
    plt.close()

#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('-n', '--tr-num', type=int, default=1)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('-f', '--frame_type', type=click.Choice(["ProtoWIB", "WIB"]),
              help="Select input frame type", default='WIB')
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default='HDColdbox')
@click.option('-t', '--threshold', type=int, default=100)
@click.option('-w', '--num-waves', type=int, default=10)
@click.option('-s', '--step', type=int, default=150)
@click.option('-o', '--outpath', help="Output path for plots", default=".")

def cli(file_path: str, tr_num : int, interactive: bool, frame_type: str, map_id: str, threshold: int, outpath: str, num_waves: int, step: int) -> None:

    dp = Path(file_path)
    outpath = Path(outpath)

    rdm = DataManager(dp.parent, frame_type, map_id)
    data_files = sorted(rdm.list_files(), reverse=True)
    rich.print(data_files)
    f = dp.name
    rich.print(f)
    trl = rdm.get_entry_list(f)
    # rich.print(trl)
    if tr_num not in trl:
        raise IndexError(f"{tr_num} does not exists!");
    
    fwtp_list = []
    tpc_list  = []
    for tr in trl:
        en_info, tpc_df, tp_df, fwtp_df = rdm.load_entry(file_path, tr)
        fwtp_list.append(fwtp_df)
        tpc_list.append(tpc_df)
    fwtp_df = pd.concat(fwtp_list)
    tpc_df  = pd.concat(tpc_list)

    fwtp_df.to_hdf(outpath  / ('fw_tp_'+ dp.stem + '.hdf5'), "fw_tp")
    return

    run = en_info['run_number']

    rich.print(tp_df.columns)

    tpgm = TPGManager(1000, "data/fir_coeffs.dat", 6, threshold, "HDColdbox")
    tp_df, ped_df, fir_df = tpgm.run_capture(tpc_df, 0, 0, pedchan=True, align=False)

    tp_df.to_hdf(outpath  / ('emu_tp_'+ dp.stem + '.hdf5'), "emu_tp")

    plt.rcParams['figure.figsize'] = [12., 5.]
    plt.rcParams['figure.dpi'] = 75
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_1d_hists_'+ dp.stem + '.pdf'))

    for var in hit_labels.keys():
        if((var == "peak_adc")or(var == "sum_adc")):
            nbins = 100
        else:
            nbins = None
        hist_plot(fwtp_df, tp_df, var, hit_labels, nbins=nbins, ylabel="# hits", pdf=pdf)

    pdf.close()

    if interactive:
        import IPython
        IPython.embed(colors="neutral")
    
if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(
       level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    cli()
