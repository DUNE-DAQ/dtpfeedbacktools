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

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return np.array([overlap_true, overlap_time])

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('-p', '--plots', is_flag=True, help="Generate a plot", default=False)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default="HDColdbox")
@click.option('-f', '--frame_type', type=click.Choice(
    [
        "ProtoWIB",
        "WIB"
    ]),
              help="Select input frame type", default="WIB")
@click.option('-o', '--outname', default="./tstamp.png")

def cli(interactive: bool, plots: bool, files_path: str, map_id: str, frame_type: str, outname: str) -> None:

    rdm = RawDataManager(files_path, frame_type, map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    rich.print(tp_files)
    rich.print(adc_files)

    t = Table()
    t.add_column("filename", style="green")
    t.add_column("link #")
    t.add_column("timestamp_min")
    t.add_column("relative offset (timestamp ticks)")
    t.add_column("capture length (timestamp ticks)")
    t.add_column("capture length (s)")
    t.add_column("overlap (s)")
    
    tstamps = -np.ones((len(tp_files)+len(adc_files), 2), dtype=int)
    links = -np.ones(len(tp_files)+len(adc_files), dtype=int)
    overlaps = -np.ones((len(tp_files)+len(adc_files), 2), dtype=int)
    file_list = []
    
    for i, f in enumerate(tp_files):
            
        file_list.append(f)

        link = 5+6*rdm.get_link(f)
        links[i] = link

        tstamps[i] = rdm.find_tp_ts_minmax(f)

        overlaps[i] = np.array([False, 0])

    for i, f in enumerate(adc_files):

        file_list.append(f)

        link = rdm.get_link(f)
        links[i+len(tp_files)] = link

        tstamps[i+len(tp_files)] = rdm.find_tpc_ts_minmax(f)

        indx = np.where(links == 5+6*int(link > 5))[0][0]
        overlaps[i+len(tp_files)] = overlap_check(tstamps[indx], tstamps[i+len(tp_files)])
        
    indx = links.argsort()
    tstamps = tstamps[indx]
    links = links[indx]
    overlaps = overlaps[indx]
    file_list = np.array(file_list, dtype=str)
    file_list = file_list[indx]
    tstamp_reference = np.min(tstamps[tstamps>0])

    for i in range(len(tstamps)):
        if links[i] == -1:
            continue
        if overlaps[i,0]:
            t.add_row(file_list[i], str(links[i]), str(tstamps[i,0]), str(tstamps[i,0]-tstamp_reference), str(tstamps[i,1]-tstamps[i,0]), str((tstamps[i,1]-tstamps[i,0])/62.5e6), str(overlaps[i,1]/62.5e6), style="red")
        else:
            t.add_row(file_list[i], str(links[i]), str(tstamps[i,0]), str(tstamps[i,0]-tstamp_reference), str(tstamps[i,1]-tstamps[i,0]), str((tstamps[i,1]-tstamps[i,0])/62.5e6), "")
            
    print(t)


    if plots:  
        fig=plt.figure(figsize=(9,7))

        tp_label = "TP capture"
        adc_label = "ADC capture"
        for i in range(len(tstamps)):
            if(links[i] in [5, 11]):
                plt.fill_between((tstamps[i]-tstamp_reference)*NS_PER_TICK*1e-9, [links[i]-.25, links[i]-.25], [links[i]+.25, links[i]+.25], color="dodgerblue", alpha=0.6, label=tp_label)
                tp_label = "_nolegend_"
            else:
                plt.fill_between((tstamps[i]-tstamp_reference)*NS_PER_TICK*1e-9, [links[i]-.25, links[i]-.25], [links[i]+.25, links[i]+.25], color="red", alpha=0.6, label=adc_label)
                adc_label = "_nolegend_"

        plt.legend(fontsize=14, loc="lower left")

        plt.xlabel("Time [s]", labelpad=10, fontsize=16)
        plt.ylabel("Link", labelpad=10, fontsize=16)
    
        plt.savefig(outpath+"time_overlap.pdf", dpi=500, bbox_inches='tight')

    
    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
