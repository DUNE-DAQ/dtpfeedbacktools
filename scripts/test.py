#!/usr/bin/env python

from rich import print as rprint

import rich
import detchannelmaps
import ctypes
import dtpfeedbacktools
import pandas as pd

from rich import print as rprint

from dtpfeedbacktools import FWTPHeader, FWTPData, FWTPTrailer

ch_map = detchannelmaps.make_map('HDColdboxChannelMap')

rawtp_path = './data/raw_record_15285/output_tp_0_0.out'
rfr = dtpfeedbacktools.RawFileReader(rawtp_path)

# Some parameters
tp_block_size = 3
tp_block_bytes = tp_block_size*4
#n_tpblocks = 100
n_tpblocks = rfr.get_size() // tp_block_bytes

rprint(f"Reading {n_tpblocks} TP blocks")
blk = rfr.read_block(size=tp_block_bytes*n_tpblocks, offset=0)

rprint(blk.size())

rprint(f"Unpacking {n_tpblocks}")
fwtps = dtpfeedbacktools.unpack_fwtps(blk.as_capsule(), n_tpblocks)
print(fwtps)

rprint(f"Loaded {len(fwtps)} FW TP packets")

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
rprint(rtp_df)

