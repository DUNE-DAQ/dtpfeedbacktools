#!/usr/bin/env python

import sys
import rich
from rich.table import Table
import logging
import time
import click
from rich import print
from rich.logging import RichHandler
from pathlib import Path
import itertools

import pandas as pd

import dtpfeedbacktools
from dtpfeedbacktools.rawdatamanager import RawDataManager, RawCaptureDetails, RawFileInfo

from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

header = ['offline_ch', 'crate_no', 'slot_no', 'fiber_no', 'wire_no', 'flags']
hit = ['start_time', 'end_time', 'peak_time', 'peak_adc', 'hit_continue', 'tp_flags', 'sum_adc']
pedinfo = ['median', 'accumulator']
all_plots = {"header":header, "hit":hit, "pedinfo":pedinfo}

NS_PER_TICK = 16
TS_CLK_FREQ=62.5e6
tp_block_size = 3
tp_block_bytes = tp_block_size*4

def get_capture_details(rdm : RawDataManager):

    link_map = {
        10: list(range(0,5)),
        11: list(range(5,10))
    }

    tp_files, adc_files = rdm.list_files()

    tp_infos = sorted([RawFileInfo(f, rdm.get_link(f)) for f in tp_files], key=lambda x: x.link_id)
    adc_infos = sorted([RawFileInfo(f, rdm.get_link(f)) for f in adc_files], key=lambda x: x.link_id)

    # Group infos by link_id
    tp_info_by_link = [ (lid,list(g)) for lid, g in itertools.groupby(tp_infos, lambda x: x.link_id) ]
    adc_info_by_link = [ (lid,list(g)) for lid, g in itertools.groupby(adc_infos, lambda x: x.link_id) ]

    # find duplicates
    tp_dups = [(lid, fis) for lid, fis in tp_info_by_link if len(fis) > 1]
    adc_dups = [(lid, fis) for lid, fis in adc_info_by_link if len(fis) > 1]

    # Throw an error if any
    if tp_dups or adc_dups:
        print("[red]Error[/red]")
        if tp_dups:
            print(f"Multiple tp captures with the same link id {tp_dups}")
        if adc_dups:
            print(f"Multiple adc captures with the same link id {adc_dups}")

    # Finally build the RawCapture object
    captures = []
    for tp in tp_infos:
        if tp.link_id not in link_map:
            raise RunTimeError(f"Link id {tp.link_id} ({tp.name}) not present in linkmap")
        cap = RawCaptureDetails()
        cap.tp_file = tp
        cap.adc_files += [ i for i in adc_infos if i.link_id in link_map[tp.link_id]]
        captures.append(cap)

    return captures


def find_boundaries(rdm, capture):
    '''Populate the capture object with min and max ts per file and calculate the overlap'''
    
    # find min and max ts for each file
    capture.tp_file.ts_min, capture.tp_file.ts_max = rdm.find_tp_ts_minmax(capture.tp_file.name)
    for f in capture.adc_files:
        f.ts_min, f.ts_max = rdm.find_tpc_ts_minmax(f.name)
        
    # find the overlap region
    ts_min = max([capture.tp_file.ts_min]+[f.ts_min for f in capture.adc_files])
    ts_max = min([capture.tp_file.ts_max]+[f.ts_max for f in capture.adc_files])

    # Set in the cap obj only if there is overlap
    if ts_max > ts_min:
        capture.ts_overlap_min, capture.ts_overlap_max = ts_min, ts_max


def capture_print(capture):
    
    ts_ov_min, ts_ov_max = capture.ts_overlap_min, capture.ts_overlap_max

    t = Table("type", "link", "file", "ts min", "ts max", "delta ts", "delta (s)", "offset (ts)", "offset (s)")
    
    # Shortcut
    ts_min, ts_max = capture.tp_file.ts_min, capture.tp_file.ts_max
    d_ts = ts_max - ts_min
    off_ts = ts_ov_min-ts_min if ts_ov_min else "-"
    off_s = off_ts/TS_CLK_FREQ if ts_ov_min else "-"
    t.add_row("tp", f"{capture.tp_file.link_id}", capture.tp_file.name, f"{ts_min}", f"{ts_max}", f"{d_ts}", f"{(d_ts)/TS_CLK_FREQ}", f"{off_ts}", f"{off_s:.6}", style="cyan")

    for f in capture.adc_files:
        d_ts = f.ts_max - f.ts_min
        off_ts = ts_ov_min- f.ts_min if ts_ov_min else "-"
        off_s = off_ts/TS_CLK_FREQ if ts_ov_min else "-"
        t.add_row("adc", f"{f.link_id}", f.name, f"{f.ts_min}", f"{f.ts_max}", f"{f.ts_max - f.ts_min}", f"{(f.ts_max - f.ts_min)/TS_CLK_FREQ}", f"{off_ts}", f"{off_s:.6}")
    # Shortcut
    if ts_ov_min and ts_ov_max:
        t.add_row("overlap", "", "", f"{ts_ov_min}", f"{ts_ov_max}", f"{ts_ov_max - ts_ov_min}", f"{(ts_ov_max - ts_ov_min)/TS_CLK_FREQ}", "", "", style="green")
    else:
        t.add_row("overlap", "", "", "-", "-", "-", "-", "", "", style="red")

    print(t)

    
def scan_tp_ts_offsets(tp_file, n_samples = 128):

    rfr = dtpfeedbacktools.RawFileReader(str(tp_file))
    max_tpblocks = rfr.get_size() // tp_block_bytes
    # max_bytes = max_tpblocks * tp_block_bytes

    # Numbr of tp_blocks per sample
    n_tpblocks = 32

    n_bytes = tp_block_bytes*n_tpblocks
    step_bytes = tp_block_bytes*(max_tpblocks-1)//n_samples

    offsets = [ i*step_bytes for i in range(n_samples)] + [rfr.get_size()-n_bytes]

    ts_offsets = []
    for i, offset in enumerate(offsets):

        blk = rfr.read_block(n_bytes, offset)

        fwtps = dtpfeedbacktools.unpack_fwtps_to_nparrays(blk.as_capsule(), n_tpblocks, safe_mode = (i != 0))
        ts = min(fwtps['ts']) if i != n_samples else min(fwtps['ts'])

        ts_offsets.append((ts, offset))

    ts_ranges = []
    for i in range(len(ts_offsets)-1):
        ts_ranges.append(ts_offsets, )


    return ts_offsets

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('--save', is_flag=True, default=False)
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
@click.option('-o', '--outdir', type=click.Path(), default=".")
@click.option('-s', '--split-factor', type=int, default=5)
@click.option('--log_level', type=click.Choice(
    [
        "DEBUG",
        "INFO",
        "CRITICAL"
    ]), help="Select log level to output", default="INFO", show_default=True)
@click.option('--log_out', is_flag=True,
              help="Redirect log info to file", default=False, show_default=True)

def cli(files_path, interactive: bool, save: bool, channel_map_name: str, frame_type: str, outdir: str, split_factor: int, log_level: str, log_out: bool) -> None:
    script = Path(__file__).stem
    if log_out:
        logging.basicConfig(
            filename=f'log_{script}_{time.strftime("%Y%m%d")}_{time.strftime("%H%M%S")}.txt',
            filemode="w",
            level=log_level,
            format="%(message)s",
            datefmt="[%X]"
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )

    capture_path = Path(files_path)
    outdir = Path(outdir)
    if not capture_path.is_dir():
        raise click.Abort("f{captuure_path} is not a directory")

    rdm = RawDataManager(str(capture_path), frame_type, channel_map_name)
    en_info = {'run_number': rdm.get_run()}
    en_info = pd.DataFrame.from_dict(en_info, orient='index').T  #just a trick to save the info into the hdf5 files
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    captures = get_capture_details(rdm)

    for c in captures:
        find_boundaries(rdm, c)

        capture_print(c)

        if not save:
            continue

        # Load TPs

        # Calculate the list of ts/byte-offsets with large granularity
        ts_offsets = scan_tp_ts_offsets(Path(rdm.data_path) / c.tp_file.name, 8192)

        #
        ts_offset_min = next(iter(i for i in range(len(ts_offsets)-1) if ts_offsets[i][0] <  c.ts_overlap_min <  ts_offsets[i+1][0]), None)
        ts_offset_max = next(iter(i+1 for i in range(len(ts_offsets)-1) if ts_offsets[i][0] <  c.ts_overlap_max <  ts_offsets[i+1][0]), None)

        # print(ts_offset_min, ts_offset_max)

        rtp_df = rdm.load_tps(
            c.tp_file.name,
            n_tpblocks=(ts_offsets[ts_offset_max][1]-ts_offsets[ts_offset_min][1])//tp_block_bytes,
            offset=ts_offsets[ts_offset_min][1]//tp_block_bytes, 
        )

        # trim
        rtp_df = rtp_df[(rtp_df["ts"] > c.ts_overlap_min) & (rtp_df["ts"] < c.ts_overlap_max) ]

        # Load ADCs
        if c.ts_overlap_max is None or c.ts_overlap_min is None:
            rich.print(f"Warning: no overlap detected for {c}. Skipping")
            continue;
        radc_dfs = []
        for f in c.adc_files:
            df = rdm.load_tpcs(
                f.name, 
                n_frames=int((c.ts_overlap_max-c.ts_overlap_min)//32), 
                offset=int((c.ts_overlap_min-f.ts_min)//32)
            )

            radc_dfs.append(df)

        radc_df = pd.concat(radc_dfs, axis=1)

        # Calculate splitting intervals
        n_bins = split_factor
        ts_edges = [c.ts_overlap_min + x * (c.ts_overlap_max- c.ts_overlap_min) // n_bins for x in range(n_bins + 1)]
        ts_bins  = [(ts_edges[k], ts_edges[k+1]) for k in range(n_bins)]

        for i, (ts_min, ts_max) in enumerate(ts_bins):

            print(f"File {i} - ts range ({ts_min}-{ts_max})")
            store = pd.HDFStore( outdir / (capture_path.name + f'_tp_link{c.tp_file.link_id}_file{i}.hdf5'))
            print("Saving run info dataframe")
            en_info.to_hdf(store, 'info')
            print("Saving raw tps dataframe")
            rtp_df[ (rtp_df['ts'] >= ts_min) & ( rtp_df['ts'] < ts_max) ].to_hdf(store, 'raw_fwtps')
            print("Saving raw adcs dataframe")
            radc_df[(radc_df.index >= ts_min) & (radc_df.index < ts_max)].to_hdf(store, 'raw_adcs')
            store.close()

    if interactive:
        import IPython
        IPython.embed(color="neutral")

if __name__ == "__main__":

    cli()
