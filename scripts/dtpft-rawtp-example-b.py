#!/usr/bin/env python

from rich import print as rprint

import rich
import detchannelmaps
import ctypes
import dtpfeedbacktools
import pandas as pd

from dtpfeedbacktools import FWTPHeader, FWTPData, FWTPTrailer

rawtp_path = './raw_record_15285/output_tp_0_0.out'

ch_map = detchannelmaps.make_map('HDColdboxChannelMap')

# rfr = dtpfeedbacktools.RawFileReader('./raw_record_15285/output_tp_0_0.out')

tp_block_size = 3
tp_block_bytes = tp_block_size*4
# # n_frames = 1024
# n_blocks = rfr.get_size() // tp_block_bytes

# blk = rfr.read_block(tp_block_bytes*n_blocks)

rawtp_buf = None
rprint(f"Opening {rawtp_path}")
with open(rawtp_path, mode="rb") as bindump:
    rawtp_buf = bindump.read()

n_blocks = len(rawtp_buf) // tp_block_bytes


rawtp_buf_p = ctypes.c_char_p(rawtp_buf)
tph_p = ctypes.cast(rawtp_buf_p, ctypes.POINTER(FWTPHeader))
tpd_p = ctypes.cast(rawtp_buf_p, ctypes.POINTER(FWTPData))
tpt_p = ctypes.cast(rawtp_buf_p, ctypes.POINTER(FWTPTrailer))


# from rich.table import Table
hdr_i = 0
trl_i = 0

fwtp_array = []

rprint(f"Unpacking {n_blocks} FW TP blocks")
# for i in range(buffer_size):
for i in range(n_blocks):
    if tpt_p[i].m_padding_1 != 0xf00d:
        continue
    trl_i = i
    tph = tph_p[hdr_i]
    tpt = tpt_p[trl_i]
    
    ts = (tph.timestamp_3 << 48) + (tph.timestamp_4 << 32) + (tph.timestamp_1 << 16) + tph.timestamp_2    
    # print(f"crate={tph.crate_no} slot={tph.slot_no} fiber={tph.fiber_no} wire={tph.wire_no} flags={tph.flags}")

    # t = Table('end_time', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'hit_continue', 'tp_flags', 'sum_adc')
    for j in range(hdr_i+1, trl_i):
        tpd = tpd_p[j]
        fwtp_array.append((
            ts,
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

    hdr_i = i+1
rprint(f"Unpacked {len(fwtp_array)} FW TPs")

rtp_df = pd.DataFrame(fwtp_array, columns=['ts', 'offline_ch', 'crate_no', 'slot_no', 'fiber_no', 'wire_no', 'flags', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'hit_continue', 'tp_flags', 'sum_adc'])

rprint(rtp_df)
import IPython
IPython.embed(colors="neutral")
