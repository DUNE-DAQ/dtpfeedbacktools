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
  m_channel:              int
  m_relative_time:        int
  m_time_over_threshold:  int
  m_adc_integral:         int

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
  tps         = []
  first       = True 
  time_shift  = 0

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
      tps.append(TP(tp.channel, relative_time, tp.time_over_threshold, tp.adc_integral))

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
  return np.array([getattr(obj, _attribute) for obj in _objects])

def MakeOnePlot(_data, _labels='title;xaxis;yaxis', _output='data',
                _plottype='hist', _min=-1, _max=-1):
  """
  Creates one plot given a detailed set of inputs. The plot can be either
  scatter, 2D histogram or 1D histogram.

  parameters:
    _data: either numpy array for 1Dhist, or a list of numpy arrays (up to 3 for coloured scatter)
    _labels: title;xaxis;yaxis labels string, the same format as root
    _output: output name. "plot_" prefix and ".png" postfix will be added automatically
    _plottype: type of plot. Either "hist", "hist2D" or "scatter"
    _min: optional minimum x-axis value (only works for 1D hist).
    _max: optional maximum x-axis value (only works for 1D hist).
  """

  print(f'Producing plot with the following data:\n  * output: {_output}\n  * plottype: {_plottype}\n  * labels: {_labels}\n')

  # Create the figure
  fplot = plt.figure()
  aplot = fplot.add_subplot(111)

  # Split the labels and set the title/axis labels
  title, xaxis, yaxis = _labels.split(';',2)
  aplot.set_xlabel(xaxis)
  aplot.set_ylabel(yaxis)
  aplot.set_title(title)

  # Add a plot
  if _plottype == 'hist':
    # Unconstrained if default
    if _min == -1:
      _min = np.min(_data)
    if _max == -1:
      _max = np.max(_data)
    aplot.hist(_data, _max - _min, (_min, _max))

  elif _plottype == 'hist2D':
    aplot.hist2d(_data[0], _data[1], bins=[100, 100])

  elif _plottype == 'scatter':
    if(len(_data) == 2):
      # No colour scatter
      aplot.scatter(_data[0], _data[1], s=2)
    else:
      # Colour scatter, added as an extra parameter
      aplot.scatter(_data[0], _data[1], s=2, c=_data[2], cmap='coolwarm', vmax=np.max(_data[2])/20)

  # Save the figure
  fplot.savefig(f'plot_{_output}.png', dpi=300)

def MakePlots(_tps):
  """
  Plots the TPs. It takes the list of TPs, converts them into numpy arrays, and
  makes various plots using the MakeOnePlot function.

  parameters:
    _tps: list of TPs
  """

  # Extract the different TP parameter numpy arrays. Probably should have used
  # with the numpy arrays rather than with this odd object to start with, so
  # that's anothe TODO...
  channelids    = GetAttributeValues(_tps, 'm_channel')
  relative_time = GetAttributeValues(_tps, 'm_relative_time')
  time_over_thr = GetAttributeValues(_tps, 'm_time_over_threshold')
  adc_integral  = GetAttributeValues(_tps, 'm_adc_integral')

  # 2D histogram of relative time vs channel id. Z axis -- TP intensity
  MakeOnePlot([relative_time, channelids],
               'TP heatmap of relative time(s) against channel ID;Channel ID;Relative time (s)',
               'channel_vs_relativetime_heatmap',
               'hist2D')

  # 2D scatter of relative time vs channel id for each TP
  MakeOnePlot([relative_time, channelids, adc_integral],
               'TP scatter of relative time(s) against channel ID;Channel ID;Relative time (s)',
               'channel_vs_relativetime_scatter',
               'scatter')

  # 1D hist of the TP intensity per channel ID
  MakeOnePlot(channelids,
              'TP frequency per Channel ID;Channel ID;Number of TPs in Channel ID',
              'tps_per_channel_hist1D',
              'hist')

  # 1D hist of the TP intensity per time over threshold
  MakeOnePlot(time_over_thr,
              'TP frequency for time over threshold;Time over threshold;Number of TPs in time over threshold',
              'tps_per_timeoverthreshold_hist1D',
              'hist')

  # 1D hist of the TP intensity per time over threshold, constrained between
  # 0--1000
  MakeOnePlot(time_over_thr,
              'TP frequency for time over threshold (constrained);Time over threshold;Number of TPs in time over threshold',
              'tps_per_timeoverthreshold_hist1D_constrained',
              'hist', 0, 1000)

  # 1D hist of the TP intensity per adc integral
  MakeOnePlot(adc_integral,
              'TP frequency per ADC Integral;ADC Integral;Number of TPs per ADC Integral bin',
              'tps_per_adcintegral_hist1D',
              'hist')

  # 1D hist of the TP intensity per adc integral, constrained between 0--400
  MakeOnePlot(adc_integral,
              'TP frequency per ADC Integral (constrained);ADC Integral;Number of TPs per ADC Integral bin',
              'tps_per_adcintegral_hist1D_constrained',
              'hist', 0, 400)

  # 2D histogram of the ADC Integral aginst Channel ID. Z axis -- TP intensity
  MakeOnePlot([adc_integral, channelids],
              'TP heatmap of ADC Integral vs Channel ID;ADC Integral;Channel ID',
              'adcintegral_vs_channelid_heatmap',
              'hist2D')

  # 2D scatter of the ADC Integral vs vs channel id for each TP
  MakeOnePlot([adc_integral, channelids],
              'TP scatter of ADC Integral vs Channel ID;ADC Integral;Channel ID',
              'adcintegral_vs_channelid_scatter',
              'scatter')

  plt.show()

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
  main(args.file, int(args.record), args.max_tps)
