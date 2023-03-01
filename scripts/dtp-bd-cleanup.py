#!/usr/bin/env python

import sys
import rich
from rich.table import Table
import os.path
import logging
import time
import click
from rich import print
from rich.logging import RichHandler
from pathlib import Path
import itertools

import numpy as np
import codecs
import matplotlib.pyplot as plt

import pandas as pd

import dtpfeedbacktools
from dtpfeedbacktools.rawdatamanager import RawDataManager, RawCaptureDetails, RawFileInfo

NS_PER_TICK = 16
TS_CLK_FREQ=62.5e6

adc_bin_map = {(0, "000"): (6,0,0), (0, "040"): (6,0,1), (0, "080"): (6,1,0), (0, "0C0"): (6,1,1), (0, "100"): (6,2,0), (0, "140"): (6,2,1),
               (1, "040"): (6,3,0), (1, "000"): (6,3,1), (1, "140"): (6,4,0), (1, "100"): (6,4,1), (1, "0C0"): (6,5,0), (1, "080"): (6,5,1)}

wib_frame_size = 118
wib_frame_bytes = wib_frame_size*4

def to_bin(i, nbits):
    return bin(i)[2:].zfill(nbits)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('--slr', type=int, default=0)
@click.option('--link', type=str, default="000")
@click.option('--nframes', type=int, default=-1)
@click.option('-m', '--channel_map_name', type=click.Choice(
    [
        "VDColdboxChannelMap",
        "HDColdboxChannelMap",
        "ProtoDUNESP1ChannelMap",
        "PD2HDChannelMap",
        "VSTChannelMap"
    ]),
    help="Select input channel map", default="HDColdboxChannelMap", show_default=True)
@click.option('-f', '--frame_type', type=click.Choice(
    [
        "ProtoWIB",
        "WIB"
    ]),
    help="Select input frame type", default="WIB")
@click.option('-o', '--outpath', type=click.Path(), default=".")
@click.option('--log_level', type=click.Choice(
    [
        "DEBUG",
        "INFO",
        "CRITICAL"
    ]), help="Select log level to output", default="INFO", show_default=True)
@click.option('--log_out', is_flag=True,
              help="Redirect log info to file", default=False, show_default=True)

def cli(files_path, interactive: bool, slr: int, link: str, nframes: int, channel_map_name: str, frame_type: str, outpath: str, log_level: str, log_out: bool) -> None:

    try:
        crate, slot, fiber = adc_bin_map[(slr, link)]
    except:
        raise ValueError
    
    dp = Path(files_path)

    header_word = hex(int(to_bin(fiber,6)+'0'+to_bin(slot,3)+to_bin(crate,10),2))[2:].zfill(5)+"0c4"
    header_word = "".join([header_word[i*2:(i+1)*2] for i in range(4)][::-1])

    rich.print(f'Header word: {header_word}')

    data = []
    with open(files_path, 'rb') as f:
        for i, chunk in enumerate(iter(lambda: f.read(4), b'')):
            string = codecs.encode(chunk, 'hex').decode("utf-8")
            data.append(string)
            if(nframes != -1)&(i > nframes): break
    data = np.array(data)

    indx = np.nonzero(data == header_word)[0]
    if len(indx) == 0:
        print("Couldn't find header word!")
        return
    chunks = [data[indx[i]:indx[i+1]] for i in range(len(indx)-1)]

    rich.print(f"Number of headers found: {len(chunks)}")

    outpath = Path(outpath)
    
    out_file = outpath  / ('out_'+ dp.stem + '.bin')
    count_written = 0
    with open(out_file, 'wb') as o:
        for i, chunk in enumerate(chunks):
            if len(chunk) != 118: continue
            count_written += 1
            for line in chunk:
                o.write(bytearray([int(line[i*2:(i*2)+2], 16) for i in range(4)]))

    rich.print(f"WIB frames written to new binary: {count_written}")

    rdm = RawDataManager(outpath, frame_type, channel_map_name)
    tpc_df = rdm.load_tpcs(out_file.name)
    tpc_df = tpc_df.reindex(sorted(tpc_df.columns), axis=1)
    rich.print(tpc_df)
    tpc_df.to_hdf(outpath  / ('out_'+ dp.stem + '.hdf5'), key="raw_adcs")

    if interactive:
        import IPython
        IPython.embed(color="neutral")

if __name__ == "__main__":

    cli()