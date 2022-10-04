#!/usr/bin/env python

# from distutils.command.install_egg_info import safe_name
from rich import print as rprint

import rich
import detchannelmaps
import ctypes
import dtpfeedbacktools
import pandas as pd
import numpy as np

from rich import print as rprint

from dtpfeedbacktools import FWTPHeader, FWTPData, FWTPTrailer

ch_map = detchannelmaps.make_map('HDColdboxChannelMap')

# Some parameters
tp_block_size = 3
tp_block_bytes = tp_block_size*4
# n_tpblocks = 100

def fwtp_list_to_df(fwtps: list):
    fwtp_array = []

    for fwtp in fwtps:
        tph = fwtp.get_header()
        tpt = fwtp.get_trailer()

        for j in range(fwtp.get_n_hits()):
            tpd = fwtp.get_data(j)
            fwtp_array.append((
                tph.get_timestamp(),
                ch_map.get_offline_channel_from_crate_slot_fiber_chan(tph.crate_no, tph.slot_no, tph.fiber_no, tph.wire_no),
                tph.crate_no, 
                tph.slot_no,
                tph.fiber_no,
                tph.wire_no,
                tph.flags,
                tpt.median,
                tpt.accumulator,
                tpd.start_time,
                tpd.end_time,
                tpd.peak_time,
                tpd.peak_adc,
                tpd.hit_continue,
                tpd.tp_flags,
                tpd.sum_adc
            ))
    rprint(f"Unpacked {len(fwtp_array)} FW TPs")

    rtp_df = pd.DataFrame(fwtp_array, columns=['ts', 'offline_ch', 'crate_no', 'slot_no', 'fiber_no', 'wire_no', 'flags', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'hit_continue', 'tp_flags', 'sum_adc'])

    return rtp_df


def unpack_entire_file(rfr):
    n_tpblocks = rfr.get_size() // tp_block_bytes

    rprint(f"Reading {n_tpblocks} TP blocks")
    blk = rfr.read_block(tp_block_bytes*n_tpblocks)


    rprint(f"Unpacking {n_tpblocks}")
    fwtps = dtpfeedbacktools.unpack_fwtps_to_nparrays(blk.as_capsule(), n_tpblocks)

    rprint(f"Building dataframe")

    # rtp_df = fwtp_list_to_df(fwtps)
    # rprint(rtp_df)
    rtp_df = pd.DataFrame(fwtps)

    rprint("Calculating offline channel numbers")
    vec_ch_map = np.vectorize(ch_map.get_offline_channel_from_crate_slot_fiber_chan)
    rtp_df['offline_ch'] = vec_ch_map(rtp_df['crate_no'], rtp_df['slot_no'], rtp_df['fiber_no'], rtp_df['wire_no'])
    return rtp_df


def find_ts_minmax(rfr):
    max_tpblocks = rfr.get_size() // tp_block_bytes
    max_bytes = max_tpblocks * tp_block_bytes

    n_tpblocks = 32
    n_bytes = tp_block_bytes*n_tpblocks
    first_blk = rfr.read_block(n_bytes)
    last_blk = rfr.read_block(n_bytes, max_bytes-n_bytes)

    first_fwtps = dtpfeedbacktools.unpack_fwtps(first_blk.as_capsule(), n_tpblocks, False)
    last_fwtps = dtpfeedbacktools.unpack_fwtps(last_blk.as_capsule(), n_tpblocks)

    first_rtp_df = fwtp_list_to_df(first_fwtps)
    last_rtp_df = fwtp_list_to_df(last_fwtps)

    rprint(f"read {len(first_rtp_df)} FWTPs at the start of file: min ts = {first_rtp_df['ts'].min()}")
    rprint(f"read {len(last_rtp_df)} FWTPs at the end of file: max ts = {first_rtp_df['ts'].max()}")


    return first_rtp_df, last_rtp_df

def scan_ts(rfr):
    max_tpblocks = rfr.get_size() // tp_block_bytes
    # max_bytes = max_tpblocks * tp_block_bytes

    n_tpblocks = 32
    n_samples = 16

    n_bytes = tp_block_bytes*n_tpblocks
    step_bytes = tp_block_bytes*(max_tpblocks-1)//n_samples

    vec_ch_map = np.vectorize(ch_map.get_offline_channel_from_crate_slot_fiber_chan)

    for i in range(n_samples):
        offset = i*step_bytes
        rprint(f"Reading block at offset {offset} {offset % tp_block_bytes}")

        blk = rfr.read_block(n_bytes, offset)

        fwtps = dtpfeedbacktools.unpack_fwtps_to_nparrays(blk.as_capsule(), n_tpblocks, safe_mode = (i != 0))

        rtp_df = pd.DataFrame(fwtps)

        rtp_df['offline_ch'] = vec_ch_map(rtp_df['crate_no'], rtp_df['slot_no'], rtp_df['fiber_no'], rtp_df['wire_no'])

        # rprint(rtp_df)
        ts_min = rtp_df['ts'].min()
        ts_max = rtp_df['ts'].max()

        rprint(f"TS range: {ts_min} -> {ts_max} : {ts_max - ts_min}")


import click

@click.command()
@click.option('-f', '--file', 'file_path', type=click.Path(exists=True), default='./raw_record_15314/output_tp_0_0.out')
@click.option('-i', '--ip', 'ipy', is_flag=True, default=False)
@click.argument('example', type=click.Choice(['unpack', 'find_ts_minmax', 'scan_ts']), default=None)
def cli(file_path, example, ipy):
    
    rfr = dtpfeedbacktools.RawFileReader(file_path)

    if example=='unpack':
        rtp_df = unpack_entire_file(rfr)

    elif example == 'find_ts_minmax':
        ts_min, ts_max = find_ts_minmax(rfr)

    elif example == 'scan_ts':
        x = scan_ts(rfr)

    if ipy:
        import IPython
        IPython.embed(colors="neutral")


if __name__ == '__main__':
    cli()