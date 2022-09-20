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
from functools import reduce

tp_block_size = 3
tp_block_bytes = tp_block_size*4

#fir_coefficients = [0,0,0,0,0,0,0,0,2,4,6,7,9,11,12,13,13,12,11,9,7,6,4,2,0,0,0,0,0,0,0,0]
#fir_correction = 64/np.linalg.norm(fir_coefficients)
fir_correction = 1

NS_PER_TICK = 16

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return np.array([overlap_true, overlap_time])

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

    rdm = RawDataManager(files_path, frame_type, map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    rich.print(tp_files)
    rich.print(adc_files)
    
    tstamps = -np.ones((len(tp_files)+len(adc_files), 2), dtype=int)
    links = -np.ones(len(tp_files)+len(adc_files), dtype=int)
    overlaps = -np.ones((len(tp_files)+len(adc_files), 2), dtype=int)
    file_list = []
    
    t = Table()
    t.add_column("adc filename", style="green")
    t.add_column("tp filename", style="green")
    t.add_column("overlap start")
    t.add_column("overlap end")

    boundaries = []

    for i, f in enumerate(tp_files):
        file_list.append(f)

        if old_format:
            link = 5+6*rdm.get_link(f)
        else:
            link = rdm.get_link(f)
        
        links[i] = link
        tstamps[i] = rdm.find_tp_ts_minmax(f)
        overlaps[i] = np.array([False, 0])

    for i, f in enumerate(adc_files):

        file_list.append(f)

        link = rdm.get_link(f)

        links[i+len(tp_files)] = link
        tstamps[i+len(tp_files)] = rdm.find_tpc_ts_minmax(f)

        if old_format:
            indx = np.where(links == 5+6*int(link > 5))[0][0]
        else:
            indx = np.where(links == 10+1*int(link > 4))[0][0]
        overlaps[i+len(tp_files)] = overlap_check(tstamps[indx], tstamps[i+len(tp_files)])

        if overlaps[i+len(tp_files), 0]:
            overlap_boundary = overlap_boundaries(tstamps[indx], tstamps[i+len(tp_files)])
            boundaries.append([file_list[i+len(tp_files)], file_list[indx]]+overlap_boundary)
            t.add_row(file_list[i+len(tp_files)], file_list[indx], str(overlap_boundary[0]), str(overlap_boundary[1]))
        
    boundaries = np.array(boundaries)

    print(t)

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

            offset_low, offset_high = rdm.linear_search_tp(tp, int(new_boundaries[j,2]), int(new_boundaries[j,3]))
            offsets[j,0] = offset_low
            offsets[j,1] = offset_high

            rtpc_temp.append(rdm.load_tpcs(new_boundaries[j,0]))

        offsets_low = np.max(offsets[:,0])
        offsets_high = np.min(offsets[:,1])

        rich.print(f'Opening TPs and ADCs in the overlap region')

        rtp_df = rdm.load_tps(tp, int((offsets_high-offsets_low)//tp_block_bytes), int(offsets_low//tp_block_bytes))
        rich.print(rtp_df)

        rtpc_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='ts'), rtpc_temp)
        rich.print(rtpc_df)

        #rtp_df.to_hdf("rtp.hdf5", key="rtp")
        #rtpc_df.to_hdf("rtpc.hdf5", key="rtp")

        plot_offset = len(rtp_df)//2

        n = 0
        pdf = matplotlib.backends.backend_pdf.PdfPages(out_path)
        for i in range(1000):
            if(n >= n_plots): break

            tstamp = rtp_df["ts"][plot_offset+i]
            channel = rtp_df["offline_ch"][plot_offset+i]
            time_start = rtp_df["start_time"][plot_offset+i]
            time_end = rtp_df["end_time"][plot_offset+i]
            time_peak = rtp_df["peak_time"][plot_offset+i]
            time_over_threshold = time_end - time_start
            adc_peak = rtp_df["peak_adc"][plot_offset+i]
            fw_median = rtp_df["median"][plot_offset+i]
            if(adc_peak < 120): continue

            mu = rtpc_df[channel].mean()
            median = rtpc_df[channel].median()
            sigma = rtpc_df[channel].std()

            adc_data = rtpc_df.loc[tstamp-32*64:tstamp+32*64*2, channel]
            adc = adc_data.values
            time = adc_data.index.astype(int) - tstamp

            wave_info = '\n'.join((
                f'{"mean = ":<18}{round(mu,2):>3}',
                f'{"std = ":<20}{round(sigma,2):>3}',
                f'{"peak adc = ":<22}{adc_peak:>3}',
                f'{"time over threshold = ":<22}{time_over_threshold*32:>3}'))

            record_info = '\n'.join((
                f'{"run number = ":<17}{rdm.get_run():>10}',
                f'{"channel = ":<17}{channel:>10}',
                f'{"tstamp = ":<9}{tstamp:>10}'))

            fig = plt.figure()
            plt.style.use('ggplot')
            ax = fig.add_subplot(111)
            props = dict(boxstyle='square', facecolor='white', alpha=0.8)
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
            ax.text(0.035, 0.95, wave_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, fontdict=mono)
            ax.text(0.60, 0.175, record_info, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props, fontdict=mono)
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
