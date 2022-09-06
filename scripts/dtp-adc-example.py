#!/usr/bin/env python


import detdataformats
import detchannelmaps
import rawdatautils.unpack
import dtpfeedbacktools
import rich
import numpy as np
import pandas as pd
import collections

from rich import print as rprint

ch_map = detchannelmaps.make_map('HDColdboxChannelMap')

adc_path = './data/output_0_0.out'
rfr = dtpfeedbacktools.RawFileReader(adc_path)

wib_frame_size = 118
wib_frame_bytes = wib_frame_size*4
# n_frames = 1024
n_frames = rfr.get_size() // wib_frame_bytes

blk = rfr.read_block(wib_frame_bytes*n_frames)

wf = detdataformats.wib2.WIB2Frame(blk.get_capsule())
wh = wf.get_header()

rprint(f"detector {wh.detector_id}, crate: {wh.crate}, slot: {wh.slot}, fibre: {wh.link}")

hdr_info = (wh.detector_id, wh.crate, wh.slot, wh.link)

off_chans = [ch_map.get_offline_channel_from_crate_slot_fiber_chan(wh.crate, wh.slot, wh.link, c) for c in range(256)]

ts = rawdatautils.unpack.wib2.np_array_timestamp_data(blk.get_capsule(), n_frames)
adcs = rawdatautils.unpack.wib2.np_array_adc_data(blk.get_capsule(), n_frames)

df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(off_chans[c], adcs[:,c]) for c in range(256)]))
df = df.set_index('ts')

rprint(df)

rprint(df.mean())
rprint(df.std())

import IPython
IPython.embed(colors="neutral")
