
import os.path
from os import walk
import fnmatch
import re
import ctypes

import detdataformats
import detchannelmaps
import rawdatautils.unpack.wib as protowib_unpack
import rawdatautils.unpack.wib2 as wib_unpack
import dtpfeedbacktools

import rich
from rich import print as rprint
import numpy as np
import pandas as pd
import collections

tp_block_size = 3
tp_block_bytes = tp_block_size*4

wib_frame_size = 118
wib_frame_bytes = wib_frame_size*4

wib_sample_in_ts = 32

def get_protowib_header_info( blk ):
        wf = detdataformats.wib.WIBFrame(blk.as_capsule())
        wh = wf.get_wib_header()

        return (0, wh.crate_no, wh.slot_no, wh.fiber_no)

def get_wib_header_info( blk ):
    wf = detdataformats.wib2.WIB2Frame(blk.as_capsule())
    wh = wf.get_header()

    return (wh.detector_id, wh.crate, wh.slot, wh.link)

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
    frametype_map = {
        'ProtoWIB': (get_protowib_header_info, protowib_unpack),
        'WIB': (get_wib_header_info, wib_unpack),
    }
    
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

    def __init__(self, data_path: str, frame_type: str = 'WIB', ch_map_id: str = 'HDColdbox') -> None:

        if not os.path.isdir(data_path):
            raise ValueError(f"Directory {data_path} does not exist")

        self.data_path = data_path
        self.ch_map_name = ch_map_id
        self.ch_map = self.make_channel_map(ch_map_id)

        self.get_hdr_info, self.blk_unpack = self.frametype_map[frame_type]

    def list_files(self) -> list:
        files = []
        for m in self.match_exprs:
            files += fnmatch.filter([item for item in next(walk(self.data_path), (None, None, []))[2] if os.path.getsize(os.path.join(self.data_path, item)) != 0], m)  # [] if no file

        tpfiles = fnmatch.filter([item for item in next(walk(self.data_path), (None, None, []))[2] if os.path.getsize(os.path.join(self.data_path, item)) != 0], self.match_tps)
        adcfiles = [file for file in files if file not in tpfiles]

        return sorted(tpfiles, reverse=True, key=lambda f: os.path.getmtime(os.path.join(self.data_path, f))), sorted(adcfiles, reverse=True, key=lambda f: os.path.getmtime(os.path.join(self.data_path, f)))

    def get_run(self):
        return [int(s) for s in self.data_path.split("_") if s.isdigit()][0]

    def get_link(self, file_name: str):
        #return max([int(s) for s in file_name.replace(".out", "").split("_") if s.isdigit()])
        return int(file_name.replace(".out", "").split("_")[-1])
    
    def fwtp_list_to_df(self, fwtps: list):
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
        #rprint(f"Unpacked {len(fwtp_array)} FW TPs")

        rtp_df = pd.DataFrame(fwtp_array, columns=['ts', 'offline_ch', 'crate_no', 'slot_no', 'fiber_no', 'wire_no', 'flags', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'hit_continue', 'tp_flags', 'sum_adc'])

        return rtp_df

    def load_tps(self, file_name: str, n_tpblocks: int = -1, offset: int = 0):
        file_path = os.path.join(self.data_path, file_name)

        rfr = dtpfeedbacktools.RawFileReader(file_path)
        #if rfr.get_size() == 0:
        #    return None

        total_tpblocks = rfr.get_size() // tp_block_bytes
        if n_tpblocks == -1:
            n_tpblocks = total_tpblocks
        if offset < 0:
            offset = total_tpblocks + offset

        #rprint(f"Number of blocks to read {n_tpblocks}")
        #rprint(f"Offset {offset}")

        rprint(f"Reading {n_tpblocks} TP blocks")
        blk = rfr.read_block(size=tp_block_bytes*n_tpblocks, offset=offset*tp_block_bytes)

        rprint(f"Unpacking {n_tpblocks}")
        fwtps = dtpfeedbacktools.unpack_fwtps(blk.as_capsule(), n_tpblocks, True)

        rprint(f"Loaded {len(fwtps)} FW TP packets")

        rtp_df = self.fwtp_list_to_df(fwtps)

        return rtp_df

    def find_tp_ts_minmax(self, file_name: str, rfr = None) -> list:
        if rfr == None:
            file_path = os.path.join(self.data_path, file_name)
            rfr = dtpfeedbacktools.RawFileReader(file_path)
        
        max_tpblocks = rfr.get_size() // tp_block_bytes
        max_bytes = max_tpblocks * tp_block_bytes

        n_tpblocks = 32
        n_bytes = tp_block_bytes*n_tpblocks
        first_blk = rfr.read_block(n_bytes)
        last_blk = rfr.read_block(n_bytes, max_bytes-n_bytes)

        first_fwtps = dtpfeedbacktools.unpack_fwtps(first_blk.as_capsule(), n_tpblocks, False)
        last_fwtps = dtpfeedbacktools.unpack_fwtps(last_blk.as_capsule(), n_tpblocks)

        first_rtp_df = self.fwtp_list_to_df(first_fwtps)
        last_rtp_df = self.fwtp_list_to_df(last_fwtps)

        #rprint(f"read {len(first_rtp_df)} FWTPs at the start of file: min ts = {first_rtp_df['ts'].min()}")
        #rprint(f"read {len(last_rtp_df)} FWTPs at the end of file: max ts = {first_rtp_df['ts'].max()}")

        return first_rtp_df['ts'].min(), last_rtp_df['ts'].max()

    def find_tpc_ts_minmax(self, file_name: str) -> list:
        file_path = os.path.join(self.data_path, file_name)
        rfr = dtpfeedbacktools.RawFileReader(file_path)

        max_frames = rfr.get_size() // wib_frame_bytes
        max_bytes = max_frames * wib_frame_bytes

        n_frames = 1
        n_bytes = wib_frame_bytes*n_frames
        first_blk = rfr.read_block(n_bytes)
        last_blk = rfr.read_block(n_bytes, max_bytes-n_bytes)

        first_ts = self.blk_unpack.np_array_timestamp_data(first_blk.as_capsule(), n_frames)[0]
        last_ts = self.blk_unpack.np_array_timestamp_data(last_blk.as_capsule(), n_frames)[-1]
        
        rich.print((last_ts-first_ts)//32*wib_frame_bytes, rfr.get_size())

        if (last_ts-first_ts)//wib_sample_in_ts*wib_frame_bytes > rfr.get_size():
            print("AAAAAA")
            n_points = 100
            prev_ts = 0
            # Switch to scanning
            frame_step = (max_frames-n_frames)//n_points
            for offset in range(0, max_frames-n_frames, frame_step):
                blk = rfr.read_block(n_bytes, offset*wib_frame_bytes)
                ts = self.blk_unpack.np_array_timestamp_data(blk.as_capsule(), n_frames)[-1]
                if prev_ts != 0 and (ts-prev_ts) != frame_step * wib_sample_in_ts:
                    last_ts = prev_ts
                    break
                print(offset, ts, ts-first_ts, last_ts-ts, ts-prev_ts)
                prev_ts = ts


            rich.print(last_ts, first_ts)
        # raise SystemExit(-1)
        
        return first_ts, last_ts


    def linear_search(self, rfr, min_bytes: int, max_bytes: int, samples: int, ts_target: int):
        sample_bytes = (max_bytes-min_bytes)//samples
        n_tpblocks = 32
        n_bytes = tp_block_bytes*n_tpblocks

        ts_first_old = 0
        ts_last_old = 0

        for i in range(samples):
            offset = min_bytes+sample_bytes*i
            first_blk = rfr.read_block(n_bytes, offset)
            last_blk = rfr.read_block(n_bytes, offset+sample_bytes)

            first_fwtps = dtpfeedbacktools.unpack_fwtps(first_blk.as_capsule(), n_tpblocks, False)
            last_fwtps = dtpfeedbacktools.unpack_fwtps(last_blk.as_capsule(), n_tpblocks)

            first_rtp_df = self.fwtp_list_to_df(first_fwtps)
            last_rtp_df = self.fwtp_list_to_df(last_fwtps)
            
            ts_first = first_rtp_df['ts'].min()
            ts_last = last_rtp_df['ts'].max()

            if((ts_first <= ts_target)&(ts_last >= ts_target)):
                return offset, ts_first, ts_last
            elif((ts_last_old <= ts_target)&(ts_first >= ts_target)):
                return offset-sample_bytes, ts_last_old, ts_first
            else:
                ts_first_old = ts_first
                ts_last_old = ts_last

    def linear_search_tp(self, file_name: str, ts_low: int, ts_high: int, n_idle: int = 2):
        file_path = os.path.join(self.data_path, file_name)
        rfr = dtpfeedbacktools.RawFileReader(file_path)

        max_tpblocks = rfr.get_size() // tp_block_bytes
        max_bytes = max_tpblocks * tp_block_bytes
        min_bytes = 0

        offset_low, ts_first_low, ts_last_low = self.linear_search(rfr, min_bytes, max_bytes, 100, ts_low)
        offset_high, ts_first_high, ts_last_high = self.linear_search(rfr, min_bytes, max_bytes, 100, ts_high)

        for i in range(n_idle - 1):
            sample_bytes = (max_bytes-min_bytes)//100
            offset_low, ts_first_low, ts_last_low = self.linear_search(rfr, offset_low, offset_low+sample_bytes, 100, ts_low)
            offset_high, ts_first_high, ts_last_high = self.linear_search(rfr, offset_high, offset_high+sample_bytes, 100, ts_high)
            max_bytes = sample_bytes
            min_bytes = 0

        rich.print(ts_first_low, ts_last_high)
        return offset_low, offset_high

    def load_tpcs(self, file_name: str, n_frames: int = -1, offset: int = 0):
        
        file_path = os.path.join(self.data_path, file_name)
        rprint(f"Loading {file_name}")
        rfr = dtpfeedbacktools.RawFileReader(file_path)

        max_frames = rfr.get_size() // wib_frame_bytes

        #if rfr.get_size() == 0:
        #    return None

        if n_frames == -1:
            n_frames = max_frames
        if offset < 0:
            offset = max_frames + offset

        rich.print(n_frames, offset)
        rich.print(n_frames*wib_frame_bytes, offset*wib_frame_bytes)

        
        blk = rfr.read_block(size=wib_frame_bytes*n_frames, offset=offset*wib_frame_bytes)

        hdr_info = self.get_hdr_info(blk)
        rprint(f"detector {hdr_info[0]}, crate: {hdr_info[1]}, slot: {hdr_info[2]}, fibre: {hdr_info[3]}")

        off_chans = [self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(hdr_info[1], hdr_info[2], hdr_info[3], c) for c in range(256)]

        #rich.print(n_frames)
        ts = self.blk_unpack.np_array_timestamp_data(blk.as_capsule(), n_frames)
        adcs = self.blk_unpack.np_array_adc_data(blk.as_capsule(), n_frames)

        rtpc_df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(off_chans[c], adcs[:,c]) for c in range(256)]))
        rtpc_df = rtpc_df.set_index('ts')
        
        return rtpc_df

