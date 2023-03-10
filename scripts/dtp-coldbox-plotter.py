#!/usr/bin/env python
# TODO: Make the output configurable
# TODO: Allow for multiple Trigger Records to be plotted on one canvas

import sys
import h5py
import numpy as np
import argparse
from dataclasses import dataclass
from matplotlib import pyplot as plt

# DAQ imports
from hdf5libs import HDF5RawDataFile
import daqdataformats
import detdataformats.trigger_primitive

@dataclass
class TP:
  m_channel: int
  m_relative_time: int

def GetRecordFromFile(_h5_file, _record):
  """
  Extracts the entire record object from a h5 file

  parameters:
    _h5_file: DAQ  hdf5libs::HDF5RawDataFile object with input HDF5
    _record: Record id to extract from that run

  return:
    A trigger record object/timeslice
  """

  # Get the number of records in the file
  records = _h5_file.get_all_record_ids()
  print(f'Number of records: {len(records)}')

  if _record > len(records):
    print(f'Asked for record no. {_record}, but only have {len(recors)} records! Exiting')
    exit()

  # Extract extract and return the record
  record = _h5_file.get_timeslice(records[_record])
  return record

def GetTPs(_record, _max_tps):
  """
  Extract the TPs from a trigger record object.

  parameters:
    _record: trigger record/timeslice object with our TPs
    _max_tps: max number of TPs to extract. -1 for all.

  return:
    list of TP dataclass objects
  """

  # Extract the fragments and get TrigerPrimitive size
  fragments = _record.get_fragments_ref()
  tp_size = detdataformats.trigger_primitive.TriggerPrimitive.sizeof()
  print(f'Number of fragments in your chosen trigger record: {len(fragments)}')

  # Initialize the list
  tps = []
  first = True 
  time_shift = 0

  # Iterate over all the fragments in the record
  for frag in fragments:
    # We're only processing TPs
    if frag.get_fragment_type() != daqdataformats.FragmentType.kTriggerPrimitive:
      continue

    # Calculate the number of frames (actual TPs) in the fragment
    hdr_size = frag.get_header().sizeof()
    n_frames = (frag.get_size() - hdr_size)//tp_size

    # Iterate over all the TPs in the fragment
    for itp in range(n_frames):
      # Get the TP as actual TriggerPrimitive object
      tp = detdataformats.trigger_primitive.TriggerPrimitive(frag.get_data(itp * tp_size))

      # Get the time shift (timestamp of the first TP)
      if first == True:
        first = False
        time_shift = tp.time_start

      # Calculate the relative time w.r.t. the first TP
      relative_time = (tp.time_start - time_shift) * 10.e-9

      # Fill the list of TP dataclass objects
      tps.append(TP(tp.channel, relative_time))

      # Keep going if we want all the TPs
      if _max_tps == -1:
        continue

      # If we don't want all the TPs, finish when we reach our maximum
      if tps.size() >= _max_tps:
        print(f'Number of tps in your chosen trigger record: {len(tps)}')
        return tps

  print(f'Number of tps in your chosen trigger record: {len(tps)}')
  return tps

def GetAttributeValues(_objects, _attribute):
  return [getattr(obj, _attribute) for obj in _objects]

def MakePlots(_tps):
  """
  Plots the TPs...
  TODO: Need to make the output configurable
  """
  # Make the plot and set the labels
  fig = plt.subplot(111)
  fig.set_xlabel("Relative time (s)")
  fig.set_ylabel("Channel ID")

  # Create a scatter plot for relative time vs channel
  fig.scatter(GetAttributeValues(_tps, 'm_relative_time'), GetAttributeValues(_tps, 'm_channel'),
      marker='.',
      )

  # TODO: Need to make the output configurable
  plt.savefig('plot_tps_channel_vs_relativetime.png')

def main(_file, _record, _max_tps):
  print(_file)
  # Get the hdf5 file
  h5_file = HDF5RawDataFile(_file)

  # Get the trigger record
  record = GetRecordFromFile(h5_file, _record)

  # Extract TPs from the record
  tps = GetTPs(record, _max_tps)

  # Make the plot
  MakePlots(tps)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="TP plotter")

  parser.add_argument('-f', '--file', dest='file', help='Input tpstream file')
  parser.add_argument('-r', '--record', dest='record', default=int(0), help='Trigger Record to look at (default 0)')
  parser.add_argument('-m', '--max_tps', dest='max_tps', default=int(-1), help='Maxumum number of TPs to plot (default -1 == all tps)')

  args = parser.parse_args()
  main(args.file, args.record, args.max_tps)
