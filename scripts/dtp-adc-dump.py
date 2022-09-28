#!/usr/bin/env python

import os.path
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
from functools import reduce

tp_block_size = 3
tp_block_bytes = tp_block_size*4

wib_frame_size = 118
wib_frame_bytes = wib_frame_size*4

#fir_coefficients = [0,0,0,0,0,0,0,0,2,4,6,7,9,11,12,13,13,12,11,9,7,6,4,2,0,0,0,0,0,0,0,0]
#fir_correction = 64/np.linalg.norm(fir_coefficients)
fir_correction = 1

NS_PER_TICK = 16

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('--old_format', is_flag=True, default=False)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default=None)
@click.option('-f', '--frame_type', type=click.Choice(
    [
        "ProtoWIB",
        "WIB"
    ]),
              help="Select input frame type", default="WIB")
@click.option('-n', '--n_plots', help="Select number of plots in output file", default=10)
@click.option('-t', '--threshold', help="Select hit threshold used", default=100)
@click.option('-o', '--out_path', help="Output path for plots", default="./validation")

def cli(interactive: bool, old_format: bool, files_path: str, map_id: str, frame_type: str, n_plots: int, threshold: int, out_path: str) -> None:

    rdm = RawDataManager(files_path, frame_type, map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    rich.print(tp_files)
    rich.print(adc_files)

    rtpc_df = rdm.load_tpcs(adc_files[0], 1000000, int(os.path.getsize(os.path.join(rdm.data_path, adc_files[0]))//wib_frame_bytes)//2)
    rich.print(rtpc_df)

    channel = rtpc_df.keys()

    pdf = matplotlib.backends.backend_pdf.PdfPages(out_path)
    for i in range(n_plots):

        mu = rtpc_df[channel[i]].mean()
        median = rtpc_df[channel[i]].median()
        sigma = rtpc_df[channel[i]].std()

        adc_data = rtpc_df[channel[i]]
        adc = adc_data.values
        time = adc_data.index.astype(int) - adc_data.index.astype(int)[0]

        wave_info = '\n'.join((
            f'{"mean = ":<7}{round(mu,2):>6}',
            f'{"std = ":<7}{round(sigma,2):>6}'))

        fig = plt.figure()
        plt.style.use('ggplot')
        ax = fig.add_subplot(111)
    
        props_wave = dict(boxstyle='square', facecolor='dodgerblue', alpha=0.3)
        mono = {'family' : 'monospace'}
        plt.plot(time, adc, c="dodgerblue", label="Raw ADC")

        ax.text(0.035, 0.95, wave_info, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props_wave, fontdict=mono)

        plt.xlim(0, 800000)

        plt.xlabel("Relative time [ticks]", fontsize=12, labelpad=10, loc="right")
        
        pdf.savefig()
        plt.close()


    pdf.close()

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
