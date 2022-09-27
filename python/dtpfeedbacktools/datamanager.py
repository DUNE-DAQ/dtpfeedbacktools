# from . watcher import Watcher
import os.path
import fnmatch
from os import walk
import re

# DUNE DAQ includes
import daqdataformats
import detdataformats.wib
import detdataformats.wib2
import detdataformats.trigger_primitive
import detchannelmaps
import hdf5libs
import rawdatautils.unpack.wib as protowib_unpack
import rawdatautils.unpack.wib2 as wib_unpack
import logging

import numpy as np
import pandas as pd
import collections
import rich
from rich import print
from itertools import groupby

"""
DataManager is responsible of raw data information management: discovery, loading, and reference runs handling

"""

def get_protowib_header_info( frag ):
        wf = detdataformats.wib.WIBFrame(frag.get_data())
        wh = wf.get_wib_header()

        logging.debug(f"detector_id {0}, crate: {wh.crate_no}, slot: {wh.slot_no}, fibre: {wh.fiber_no}")

        return (0, wh.crate_no, wh.slot_no, wh.fiber_no)

def get_wib_header_info( frag ):
    wf = detdataformats.wib2.WIB2Frame(frag.get_data())
    wh = wf.get_header()
    logging.debug(f"detector {wh.detector_id}, crate: {wh.crate}, slot: {wh.slot}, fibre: {wh.link}")

    return (wh.detector_id, wh.crate, wh.slot, wh.link)

class VSTChannelMap(object):

    @staticmethod
    def get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, fiber_no, ch_no):
        return 256*fiber_no+ch_no

    @staticmethod
    def get_plane_from_offline_channel(ch):
        return 0


class DataManager:

    # match_exprs = ['*.hdf5','*.hdf5.copied']
    match_exprs = ['*.hdf5', '*.hdf5.copied']
    max_cache_size = 100
    frametype_map = {
        'ProtoWIB': (get_protowib_header_info, protowib_unpack, daqdataformats.FragmentType.kProtoWIB),
        'WIB': (get_wib_header_info, wib_unpack, daqdataformats.FragmentType.kWIB),
    }

    @staticmethod 
    def make_channel_map(map_id):

        if map_id == 'VDColdbox':
            return detchannelmaps.make_map('VDColdboxChannelMap')
        elif map_id == 'ProtoDUNESP1':
            return detchannelmaps.make_map('ProtoDUNESP1ChannelMap')
        elif map_id == 'PD2HD':
            return detchannelmaps.make_map('PD2HDChannelMap')
        elif map_id == 'VST':
            return VSTChannelMap()
        elif map_id == 'HDColdbox':
            return detchannelmaps.make_map('HDColdboxChannelMap')
        else:
            raise RuntimeError(f"Unknown channel map id '{map_id}'")


    def __init__(self, data_path: str, frame_type: str = 'ProtoWIB', ch_map_id: str = 'VDColdbox') -> None:

        if not os.path.isdir(data_path):
            raise ValueError(f"Directory {data_path} does not exist")

        if frame_type not in ['ProtoWIB', "WIB"]:
            raise ValueError(f"Unknown fragment type {frame_type}")

        logging.warning(f"Frame type: {frame_type}")
        self.data_path = data_path
        self.ch_map_name = ch_map_id
        self.ch_map = self.make_channel_map(ch_map_id) 

        self.offch_to_hw_map = self._init_o2h_map()
        self.femb_to_offch = {k: [int(x) for x in d] for k, d in groupby(self.offch_to_hw_map, self.femb_id_from_offch)}

        # self.trig_rec_hdr_regex = re.compile(r"\/\/TriggerRecord(\d{5})\/TriggerRecordHeader")
        self.cache = collections.OrderedDict()
        self.get_hdr_info, self.frag_unpack, self.frag_type = self.frametype_map[frame_type]

    def _init_o2h_map(self):
        if self.ch_map_name == 'VDColdbox':
            crate_no = 4
            slots = range(4)
            fibres = range(1, 3)
            chans = range(256)
        else:
            return {}

        o2h_map = {}
        for slot_no in slots:
            for fiber_no in fibres:
                for ch_no in chans:
                    off_ch = self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, fiber_no, ch_no)
                    if off_ch == 4294967295:
                        continue
                    o2h_map[off_ch] = (crate_no, slot_no, fiber_no, ch_no)

        return o2h_map

    def femb_id_from_offch(self, off_ch):
        # off_ch_str = str(off_ch)
        crate, slot, link, ch = self.offch_to_hw_map[off_ch]
        return (4*slot+2*(link-1)+ch//128)+1 


    def list_files(self) -> list:
        files = []
        for m in self.match_exprs:
            files += fnmatch.filter(next(walk(self.data_path), (None, None, []))[2], m)  # [] if no file

        return sorted(files, reverse=True, key=lambda f: os.path.getmtime(os.path.join(self.data_path, f)))

    def has_trigger_records(self, file_name: str) -> list:
        file_path = os.path.join(self.data_path, file_name)
        rdf = hdf5libs.HDF5RawDataFile(file_path) # number of events = 10000 is not used
        try:
            _ = rdf.get_all_trigger_record_ids()
            return True
        except RuntimeError:
            return False

    def has_timeslices(self, file_name: str) -> list:
        file_path = os.path.join(self.data_path, file_name)
        rdf = hdf5libs.HDF5RawDataFile(file_path) # number of events = 10000 is not used
        try:
            return [ n for n,_ in rdf.get_all_timeslice_ids()]
        except RuntimeError:
            return []

    def get_trigger_record_list(self, file_name: str) -> list:
        file_path = os.path.join(self.data_path, file_name)
        rdf = hdf5libs.HDF5RawDataFile(file_path) # number of events = 10000 is not used
        try:
            return [ n for n,_ in rdf.get_all_trigger_record_ids()]
        except RuntimeError:
            return []

    def get_timeslice_list(self, file_name: str) -> list:
        file_path = os.path.join(self.data_path, file_name)
        rdf = hdf5libs.HDF5RawDataFile(file_path) # number of events = 10000 is not used
        try:
            return [ n for n,_ in rdf.get_all_timeslice_ids()]
        except RuntimeError:
            return []


    def get_entry_list(self, file_name: str) -> list:
        trl = self.get_trigger_record_list(file_name)
        tsl = self.get_timeslice_list(file_name)

        return trl if trl else tsl


    def load_entry(self, file_name: str, entry: int) -> list:
        uid = (file_name, entry)
        if uid in self.cache:
            logging.info(f"{file_name}:{entry} already loaded. returning cached dataframe")
            e_info, tpc_df, tp_df = self.cache[uid]
            self.cache.move_to_end(uid, False)
            return e_info, tpc_df, tp_df

        file_path = os.path.join(self.data_path, file_name)
        rdf = hdf5libs.HDF5RawDataFile(file_path) # number of events = 10000 is not used
            
        # Check if we're dealing with tr ot ts
        has_trs = False
        try:
            _ = rdf.get_all_trigger_record_ids()
            has_trs = True
        except:
            pass

        has_tss = False
        try:
            _ = rdf.get_all_timeslice_ids()
            has_tss = True
        except:
            pass

        #----

        if has_trs:
            logging.debug(f"Trigger Records detected!")
            get_ehdr = rdf.get_trh
        elif has_tss:
            logging.debug(f"TimeSlices detected!")
            get_ehdr = rdf.get_tsh

        else:
            raise RuntimeError(f"No TriggerRecords nor TimeSlices found in {file_name}")

        en_hdr = get_ehdr((entry,0))
        en_source_ids = rdf.get_source_ids((entry, 0))

        if has_trs:
            en_info = {
                'run_number': en_hdr.get_run_number(),
                'trigger_number': en_hdr.get_trigger_number(),
                'trigger_timestamp': en_hdr.get_trigger_timestamp(),
            }
            en_ts = en_hdr.get_trigger_timestamp()

        elif has_tss:
            en_info = {
                'run_number': en_hdr.run_number,
                'trigger_number': en_hdr.timeslice_number,
                'trigger_timestamp': 0,
            }
            en_ts = 0

        print(en_info)

        tpc_dfs = []
        tp_array = []
        for sid in en_source_ids:
            frag = rdf.get_frag((entry, 0),sid)
            frag_hdr = frag.get_header()

            logging.debug(f"Inspecting {sid.version}, {sid.subsystem}, {sid.id}")
            logging.debug(f"Run number : {frag.get_run_number()}")
            logging.debug(f"Trigger number : {frag.get_trigger_number()}")
            logging.debug(f"Trigger TS    : {frag.get_trigger_timestamp()}")
            logging.debug(f"Window begin  : {frag.get_window_begin()}")
            logging.debug(f"Window end    : {frag.get_window_end()}")
            logging.debug(f"Fragment type : {frag.get_fragment_type()}")
            logging.debug(f"Fragment code : {frag.get_fragment_type_code()}")
            logging.debug(f"Size          : {frag.get_size()}")

            if (sid.subsystem == daqdataformats.SourceID.kDetectorReadout and frag.get_fragment_type() == self.frag_type):
                payload_size = (frag.get_size()-frag_hdr.sizeof())
                if not payload_size:
                    continue
                rich.print(f"Number of WIB frames: {payload_size}")

                det_id, crate_no, slot_no, link_no = self.get_hdr_info(frag)
                logging.debug(f"crate: {crate_no}, slot: {slot_no}, fibre: {link_no}")

                off_chans = [self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, link_no, c) for c in range(256)]

                ts = self.frag_unpack.np_array_timestamp(frag)
                adcs = self.frag_unpack.np_array_adc(frag)
                #ts = (ts - en_ts).astype('int64')
                logging.debug(f"Unpacking {sid.version}, {sid.subsystem}, {sid.id} completed")

                df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(off_chans[c], adcs[:,c]) for c in range(256)]))
                df = df.set_index('ts')
                # rich.print(df)

                tpc_dfs.append(df)

            elif frag.get_fragment_type() == daqdataformats.FragmentType.kSW_TriggerPrimitive:
                tp_size = detdataformats.trigger_primitive.TriggerPrimitive.sizeof()
                n_frames = (frag.get_size()-frag_hdr.sizeof())//tp_size
                rich.print(f"Number of TPS frames: {n_frames}")
                for i in range(n_frames):
                    tp = detdataformats.trigger_primitive.TriggerPrimitive(frag.get_data(i*tp_size))
                    # tp_array.append( (tp.time_peak-en_ts, tp.channel, tp.adc_peak) )
                    #tp_array.append( (tp.time_start-en_ts, tp.time_peak-en_ts, tp.time_over_threshold, tp.channel, tp.adc_integral, tp.adc_peak, tp.flag) )
                    tp_array.append( (tp.time_start, tp.time_peak, tp.time_over_threshold, tp.channel, tp.adc_integral, tp.adc_peak, tp.flag) )
                    
        tp_df = pd.DataFrame(tp_array, columns=['time_start', 'time_peak', 'time_over_threshold', 'channel', 'adc_integral', 'adc_peak', 'flag'])
        
        if tpc_dfs:
            tpc_df = pd.concat(tpc_dfs, axis=1)
            # Sort columns (channels)
            tpc_df = tpc_df.reindex(sorted(tpc_df.columns), axis=1)
        else:
            tpc_df = pd.DataFrame( columns=['ts'])
            tpc_df = tpc_df.set_index('ts')

        self.cache[uid] = (en_info, tpc_df, tp_df)
        if len(self.cache) > self.max_cache_size:
            old_uid, _ = self.cache.popitem(False)
            logging.info(f"Removing {old_uid[0]}:{old_uid[1]} from cache")

        return en_info, tpc_df, tp_df

    def load_trigger_record(self, file_name: str, tr_num: int) -> list:

        uid = (file_name, tr_num)
        if uid in self.cache:
            logging.info(f"{file_name}:{tr_num} already loaded. returning cached dataframe")
            tr_info, tr_df, tp_df = self.cache[uid]
            self.cache.move_to_end(uid, False)
            return tr_info, tr_df, tp_df

        file_path = os.path.join(self.data_path, file_name)
        rdf = hdf5libs.HDF5RawDataFile(file_path) # number of events = 10000 is not used

        tr_hdr = rdf.get_trh((tr_num,0))
        tr_source_ids = rdf.get_source_ids((tr_num, 0))


        tr_info = {
            'run_number': tr_hdr.get_run_number(),
            'trigger_number': tr_hdr.get_trigger_number(),
            'trigger_timestamp': tr_hdr.get_trigger_timestamp(),
        }

        tr_ts = tr_hdr.get_trigger_timestamp()


        tpc_dfs = []
        tp_array = []
        for sid in tr_source_ids:
            frag = rdf.get_frag((tr_num, 0),sid)
            frag_hdr = frag.get_header()

            logging.debug(f"Inspecting {sid.version}, {sid.subsystem}, {sid.id}")
            logging.debug(f"Run number : {frag.get_run_number()}")
            logging.debug(f"Trigger number : {frag.get_trigger_number()}")
            logging.debug(f"Trigger TS    : {frag.get_trigger_timestamp()}")
            logging.debug(f"Window begin  : {frag.get_window_begin()}")
            logging.debug(f"Window end    : {frag.get_window_end()}")
            logging.debug(f"Fragment type : {frag.get_fragment_type()}")
            logging.debug(f"Fragment code : {frag.get_fragment_type_code()}")
            logging.debug(f"Size          : {frag.get_size()}")

            if sid.subsytem ==  self.frag_type:
                payload_size = (frag.get_size()-frag_hdr.sizeof())
                if not payload_size:
                    continue

                det_id, crate_no, slot_no, link_no = self.get_hdr_info(frag)
                logging.debug(f"crate: {crate_no}, slot: {slot_no}, fibre: {link_no}")

                off_chans = [self.ch_map.get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, link_no, c) for c in range(256)]

                ts = self.frag_unpack.np_array_timestamp(frag)
                adcs = self.frag_unpack.np_array_adc(frag)
                ts = (ts - tr_ts).astype('int64')
                logging.debug(f"Unpacking {sid.version}, {sid.subsystem}, {sid.id} completed")

                df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(off_chans[c], adcs[:,c]) for c in range(256)]))
                df = df.set_index('ts')
                # rich.print(df)

                tpc_dfs.append(df)

            elif sid.subsystem == daqdataformats.SourceID.kTrigger and frag.get_fragment_type() == daqdataformats.FragmentType.kSW_TriggerPrimitive:
                tp_size = detdataformats.trigger_primitive.TriggerPrimitive.sizeof()
                n_frames = (frag.get_size()-frag_hdr.sizeof())//tp_size
                rich.print(f"Number of TPS frames: {n_frames}")
                for i in range(n_frames):
                    tp = detdataformats.trigger_primitive.TriggerPrimitive(frag.get_data(i*tp_size))
                    tp_array.append( (tp.time_peak-tr_ts, tp.channel, tp.adc_peak) )



        tp_df = pd.DataFrame(tp_array, columns=['time', 'channel', 'adc_peak'])
        # rich.print(tp_array)
        tpc_df = pd.concat(tpc_dfs, axis=1)
        # Sort columns (channels)
        tpc_df = tpc_df.reindex(sorted(tpc_df.columns), axis=1)

        self.cache[uid] = (tr_info, tpc_df, tp_df)
        if len(self.cache) > self.max_cache_size:
            old_uid, _ = self.cache.popitem(False)
            logging.info(f"Removing {old_uid[0]}:{old_uid[1]} from cache")

        return tr_info, tpc_df, tp_df
