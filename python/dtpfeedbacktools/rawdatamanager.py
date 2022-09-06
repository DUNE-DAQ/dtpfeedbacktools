
import os.path
from os import walk
import fnmatch
import re
import ctypes

import detdataformats
import detchannelmaps
import rawdatautils.unpack
import dtpfeedbacktools
from dtpfeedbacktools import FWTPHeader, FWTPData, FWTPTrailer

import rich
from rich import print as rprint
import numpy as np
import pandas as pd
import collections

tp_block_size = 3
tp_block_bytes = tp_block_size*4

wib_frame_size = 118
wib_frame_bytes = wib_frame_size*4

class VSTChannelMap(object):

    @staticmethod
    def get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, fiber_no, ch_no):
        return 256*fiber_no+ch_no

    @staticmethod
    def get_plane_from_offline_channel(ch):
        return 0

class RawDataManager:
    
    match_exprs = ['*.out']
    match_tps = '*_tp_*'
    
    @staticmethod 
    def make_channel_map(map_id):

        if map_id == 'VDColdbox':
            return detchannelmaps.make_map('VDColdboxChannelMap')
        elif map_id == 'HDColdbox':
            return detchannelmaps.make_map('HDColdboxChannelMap')
        elif map_id == 'ProtoDUNESP1':
            return detchannelmaps.make_map('ProtoDUNESP1ChannelMap')
        elif map_id == 'PD2HD':
            return detchannelmaps.make_map('PD2HDChannelMap')
        elif map_id == 'VST':
            return VSTChannelMap()
        else:
            raise RuntimeError(f"Unknown channel map id '{map_id}'")

    def __init__(self, data_path: str, ch_map_id: str = 'HDColdbox') -> None:

        if not os.path.isdir(data_path):
            raise ValueError(f"Directory {data_path} does not exist")

        self.data_path = data_path
        self.ch_map_name = ch_map_id
        self.ch_map = self.make_channel_map(ch_map_id) 

    def list_files(self) -> list:
        files = []
        for m in self.match_exprs:
            files += fnmatch.filter(next(walk(self.data_path), (None, None, []))[2], m)  # [] if no file

        tpfiles = fnmatch.filter(next(walk(self.data_path), (None, None, []))[2], self.match_tps)
        adcfiles = [file for file in files if file not in tpfiles]

        return sorted(tpfiles, reverse=True, key=lambda f: os.path.getmtime(os.path.join(self.data_path, f))), sorted(adcfiles, reverse=True, key=lambda f: os.path.getmtime(os.path.join(self.data_path, f)))

    def get_link(self, file_name: str):
        return max([int(s) for s in file_name.replace(".out", "").split("_") if s.isdigit()])
        
    def load_tps(self, file_name: str, n_tpblocks: int = -1, offset: int = 0):
        file_path = os.path.join(self.data_path, file_name)

        rfr = dtpfeedbacktools.RawFileReader(file_path)
        total_tpblocks = rfr.get_size() // tp_block_bytes
        if n_tpblocks == -1:
            n_tpblocks = total_tpblocks
        if offset < 0:
            offset = total_tpblocks + offset

        rprint(f"Reading {n_tpblocks} TP blocks")
        blk = rfr.read_block(size=tp_block_bytes*n_tpblocks, offset=offset*tp_block_bytes)

        rprint(f"Unpacking {n_tpblocks}")
        fwtps = dtpfeedbacktools.unpack_fwtps(blk.get_capsule(), n_tpblocks, True)

        rprint(f"Loaded {len(fwtps)} FW TP packets")

        fwtp_array = []

        for fwtp in fwtps:
            tph = fwtp.get_header()
            tpt = fwtp.get_trailer()

            for j in range(fwtp.get_n_hits()):
                tpd = fwtp.get_data(j)
                fwtp_array.append((
                    tph.get_timestamp(),
                    self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(tph.crate_no, tph.slot_no, tph.fiber_no, tph.wire_no),
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

    def load_tpcs(self, file_name: str):
        
        file_path = os.path.join(self.data_path, file_name)
        rprint(f"Opening {file_name}")
        rfr = dtpfeedbacktools.RawFileReader(file_path)

        n_frames = rfr.get_size() // wib_frame_bytes

        if rfr.get_size() == 0:
            return None
        
        blk = rfr.read_block(wib_frame_bytes*n_frames)

        wf = detdataformats.wib2.WIB2Frame(blk.get_capsule())
        wh = wf.get_header()

        rprint(f"detector {wh.detector_id}, crate: {wh.crate}, slot: {wh.slot}, fibre: {wh.link}")

        hdr_info = (wh.detector_id, wh.crate, wh.slot, wh.link)

        off_chans = [self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(wh.crate, wh.slot, wh.link, c) for c in range(256)]

        ts = rawdatautils.unpack.wib2.np_array_timestamp_data(blk.get_capsule(), n_frames)
        adcs = rawdatautils.unpack.wib2.np_array_adc_data(blk.get_capsule(), n_frames)

        rtpc_df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(off_chans[c], adcs[:,c]) for c in range(256)]))
        rtpc_df = rtpc_df.set_index('ts')
        
        return rtpc_df
