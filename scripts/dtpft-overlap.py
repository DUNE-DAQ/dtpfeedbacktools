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

NS_PER_TICK = 16

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return np.array([overlap_true, overlap_time])

def overlap_boundaries(tp_tstamp, adc_tstamp):
    return np.array([max(tp_tstamp[0], adc_tstamp[0]), min(tp_tstamp[1], adc_tstamp[1])])

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-p', '--plots', is_flag=True, default=False)
@click.option('-i', '--interactive', is_flag=True, default=False)
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
@click.option('-o', '--outpath', default="./validation")

def cli(interactive: bool, plots: bool, files_path: str, map_id: str, outpath: str) -> None:

    rdm = RawDataManager(files_path, map_id)
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

        if overlaps[i+len(tp_files), 0]:
            boundaries = overlap_boundaries(tstamps[indx], tstamps[i+len(tp_files)])
            t.add_row(file_list[i+len(tp_files)], file_list[indx], str(boundaries[0]), str(boundaries[1]))

    print(t)

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
