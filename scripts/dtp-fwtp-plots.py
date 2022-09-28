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

fir_correction = 1

NS_PER_TICK = 16

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return overlap_true, overlap_time

def overlap_boundaries(tp_tstamp, adc_tstamp):
    return [max(tp_tstamp[0], adc_tstamp[0]), min(tp_tstamp[1], adc_tstamp[1])]

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

    rdm = RawDataManager(files_path, frame_type, map_id, old_format)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    rich.print(tp_files)
    rich.print(adc_files)

    t, overlap_summary_df = rdm.check_overlap()

    rich.print(overlap_summary_df)
    return


    overlap_tps = np.unique(boundaries[:,1])
    rich.print(overlap_tps)

    #rtpc0_df = rdm.load_tpcs(boundaries[0,0])
    #rtpc1_df = rdm.load_tpcs(boundaries[1,0])

    for tp in overlap_tps:
        new_boundaries = boundaries[boundaries[:,1] == tp]
        rich.print(new_boundaries)

        offsets = -np.ones((len(new_boundaries),2))
        rtpc_temp = []
        for j in range(len(new_boundaries)):

            offset_low, offset_high = rdm.linear_search_tp(tp, int(new_boundaries[j,2]), int(new_boundaries[j,3]), 4)
            offsets[j,0] = offset_low
            offsets[j,1] = offset_high

            rich.print(new_boundaries[j,0])
            rtpc_temp.append(rdm.load_tpcs(new_boundaries[j,0], int(10e5), int(os.path.getsize(os.path.join(rdm.data_path, new_boundaries[j,0]))//wib_frame_bytes)//2))
            #rtpc_temp.append(rdm.load_tpcs(new_boundaries[j,0]))

        offsets_low = np.max(offsets[:,0])
        offsets_high = np.min(offsets[:,1])

        rich.print(f'Opening TPs and ADCs in the overlap region')

        #rtp_df = rdm.load_tps(tp, int((offsets_high-offsets_low)//tp_block_bytes), int(offsets_low//tp_block_bytes))
        rtp_df = rdm.load_tps(tp, int(tp_block_bytes*100000), int(offsets_low//tp_block_bytes)+int((offsets_high-offsets_low)//tp_block_bytes)//16)
        rich.print(rtp_df)

        rtpc_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='ts'), rtpc_temp)
        rich.print(rtpc_df)

        #rtp_df.to_hdf("rtp.hdf5", key="rtp")
        #rtpc_df.to_hdf("rtpc.hdf5", key="rtp")

        plot_offset = np.nonzero((rtp_df["ts"].values > rtpc_df.index[0])&(rtp_df["ts"].values < rtpc_df.index[-1]))[0][0]+1
        #plot_offset = len(rtp_df)//6
        plot_offset += 100000
        zoom_extra_range = 32*64*16
        n = 0
        pdf = matplotlib.backends.backend_pdf.PdfPages(out_path)
        for i in range(1000):
            if(n >= n_plots): break

            tstamp = rtp_df["ts"][plot_offset+i]
            if(i == 0):
                rich.print(tstamp)
            channel = rtp_df["offline_ch"][plot_offset+i]
            time_start = rtp_df["start_time"][plot_offset+i]
            time_end = rtp_df["end_time"][plot_offset+i]
            time_peak = rtp_df["peak_time"][plot_offset+i]
            time_over_threshold = time_end - time_start
            adc_peak = rtp_df["peak_adc"][plot_offset+i]
            fw_median = rtp_df["median"][plot_offset+i]
            accumulator = rtp_df["accumulator"][plot_offset+i]
            if(adc_peak < 120): continue

            mu = rtpc_df[channel].mean()
            median = rtpc_df[channel].median()
            sigma = rtpc_df[channel].std()

            adc_data = rtpc_df.loc[tstamp-zoom_extra_range:tstamp+zoom_extra_range, channel]

            adc = adc_data.values
            time = adc_data.index.astype(int) - tstamp

            wave_info = '\n'.join((
                f'{"mean = ":<7}{round(mu,2):>6}',
                f'{"std = ":<7}{round(sigma,2):>6}'))

            tp_info = '\n'.join((
                f'{"median = ":<14}{fw_median:>4}',
                f'{"accumulator = ":<14}{accumulator:>4}',
                f'{"peak adc = ":<14}{adc_peak:>4}',
                f'{"tot [tt] = ":<14}{time_over_threshold:>4}'))

            record_info = '\n'.join((
                f'{"run number = ":<17}{rdm.get_run():>10}',
                f'{"channel = ":<17}{channel:>10}',
                f'{"tstamp = ":<9}{tstamp:>10}'))

            fig = plt.figure()
            plt.style.use('ggplot')
            ax = fig.add_subplot(111)
            props_record = dict(boxstyle='square', facecolor='white', alpha=0.8)
            props_tp = dict(boxstyle='square', facecolor='red', alpha=0.3)
            props_wave = dict(boxstyle='square', facecolor='dodgerblue', alpha=0.3)
            mono = {'family' : 'monospace'}
            plt.plot(time, adc, c="dodgerblue", label="Raw ADC")
            for i in range(4):
                plt.axvline(x=-2048+i*2048, linestyle="--", c="k", alpha=0.2)
            plt.axvspan(time_start*32, time_end*32, alpha=0.2, color='red')
            plt.axvspan((time_start-15)*32, (time_end-15)*32, alpha=0.4, color='red')
            plt.axvline(x=time_peak*32, linestyle="--", c="k", alpha=0.3)
            plt.axvline(x=(time_peak-15)*32, linestyle="--", c="k", alpha=0.6)
            ax.hlines(y=fw_median, xmin=0, xmax=2048, linestyle="-.", colors="black", alpha=0.5, label="median")
            ax.hlines(y=fw_median+threshold*fir_correction, xmin=0, xmax=2048, linestyle="-.", colors="limegreen", alpha=0.5, label="median+threshold")
            ax.text(0.035, 0.95, wave_info, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props_wave, fontdict=mono)
            ax.text(0.035, 0.175, tp_info, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props_tp, fontdict=mono)
            ax.text(0.60, 0.175, record_info, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props_record, fontdict=mono)
            plt.ylim(median-300, median+300)
            plt.xlabel("Relative time [ticks]", fontsize=12, labelpad=10, loc="right")
            legend = plt.legend(fontsize=8, loc="upper right")
            frame = legend.get_frame()
            frame.set_color('white')
            frame.set_alpha(0.8)
            frame.set_linewidth(0)
            pdf.savefig()
            plt.close()

            n += 1

        pdf.close()

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
