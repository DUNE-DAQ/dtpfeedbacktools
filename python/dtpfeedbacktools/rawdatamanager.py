
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
        
    def load_tps(self, file_name: str):
        file_path = os.path.join(self.data_path, file_name)
        rawtp_buf = None
        rprint(f"Opening {file_name}")
        with open(file_path, mode="rb") as bindump:
            rawtp_buf = bindump.read()

        if len(rawtp_buf) == 0:
            return None
            
        n_blocks = len(rawtp_buf) // tp_block_bytes

        rawtp_buf_p = ctypes.c_char_p(rawtp_buf)
        tph_p = ctypes.cast(rawtp_buf_p, ctypes.POINTER(FWTPHeader))
        tpd_p = ctypes.cast(rawtp_buf_p, ctypes.POINTER(FWTPData))
        tpt_p = ctypes.cast(rawtp_buf_p, ctypes.POINTER(FWTPTrailer))

        hdr_i = 0
        trl_i = 0

        fwtp_array = []

        rprint(f"Unpacking {n_blocks} FW TP blocks")

        for i in range(n_blocks):
            if tpt_p[i].m_padding_1 != 0xf00d:
                continue
            trl_i = i
            tph = tph_p[hdr_i]
            tpt = tpt_p[trl_i]
    
            ts = (tph.timestamp_3 << 48) + (tph.timestamp_4 << 32) + (tph.timestamp_1 << 16) + tph.timestamp_2

            for j in range(hdr_i+1, trl_i):
                tpd = tpd_p[j]
                fwtp_array.append((
                    ts,
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

            hdr_i = i+1
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

        wf = detdataformats.wib2.WIB2Frame(blk.as_capsule())
        wh = wf.get_header()

        rprint(f"detector {wh.detector_id}, crate: {wh.crate}, slot: {wh.slot}, fibre: {wh.link}")

        hdr_info = (wh.detector_id, wh.crate, wh.slot, wh.link)

        off_chans = [self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(wh.crate, wh.slot, wh.link, c) for c in range(256)]

        ts = rawdatautils.unpack.wib2.np_array_timestamp_data(blk.as_capsule(), n_frames)
        adcs = rawdatautils.unpack.wib2.np_array_adc_data(blk.as_capsule(), n_frames)

        rtpc_df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(off_chans[c], adcs[:,c]) for c in range(256)]))
        rtpc_df = rtpc_df.set_index('ts')
        
        return rtpc_df
