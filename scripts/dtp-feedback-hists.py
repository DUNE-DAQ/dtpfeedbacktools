#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
import sys
import rich
import logging
import click
from rich import print
from rich.logging import RichHandler
import time
from pathlib import Path
from pylab import cm
from matplotlib import colors

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf

plt.style.use("ggplot")

fir_shift = 15

header_labels = {'crate_no':("crate no", ""), 'slot_no':("slot no", ""), 'fiber_no':("fiber no", ""), 'wire_no':("wire no", ""),
                 'start_time':("start time", " [ticks]"), 'end_time':("end time", " [ticks]"), 'peak_time':("peak time", " [ticks]"),
                 'peak_adc':("peak adc", " [ADCs]"), 'hit_continue':("hit continue", ""), 'sum_adc':("sum adc", " [adc]")}

CLK_FREQUENCY = 62.5e6
TS_PER_WIB = 32
WIB_PER_TR = 8192

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def rms(array):
    return np.sqrt(np.sum(np.power(array.astype(int), 2))/len(array.astype(int)))

# NOTE: this shouldn't be here, will move to datamanager eventually
hw_map_paths = {"APA1": "data/np04_hw_map_APA1.txt", "APA2": "data/np04_hw_map_APA2.txt"}

def open_hw_map(hw_map_name):
    hw_map_path = hw_map_paths[hw_map_name]
    hw_map_df = pd.read_csv(hw_map_path, index_col=False, header=1, delimiter=" ", names=["DRO_SourceID", "DetLink", "DetSlot", "DetCrate", "DetID", "DRO_Host", "DRO_Card", "DRO_SLR", "DRO_Link"])
    hw_map = {}
    for i, line in hw_map_df.iterrows():
        hw_map[(line.DetCrate, line.DetSlot, line.DetLink)] = line.DRO_Link+6*line.DRO_SLR
    return hw_map

def get_link(fwtp, hw_map):
    try:
        return hw_map[(fwtp.crate_no, fwtp.slot_no, fwtp.fiber_no)]
    except:
        return -1
    
def get_fwtp_rates(fwtp_df):
    tr_nums = fwtp_df["trigger_number"].unique()
    fwtp_rates = [fwtp_df.loc[fwtp_df["trigger_number"] == trigger].shape[0]/(WIB_PER_TR*TS_PER_WIB/CLK_FREQUENCY)*1e-6 for trigger in tr_nums]
    return tr_nums, fwtp_rates

def fwtp_rates_plot(fwtp_df, dp, outpath):
    links = fwtp_df["link_no"].unique()

    fig = plt.figure()

    for link in links:
        tr_nums, fwtp_rates = get_fwtp_rates(fwtp_df.loc[fwtp_df["link_no"] == link])
        #plt.step(tr_nums, fwtp_rates, where="mid", label=f'{link}')
        plt.plot(tr_nums, fwtp_rates, "--", alpha=0.4, marker="v", label=f'{link}')
    
    plt.xlabel(f'Trigger Number', labelpad=20, fontsize=16)
    plt.ylabel(r'FWTP rate [MHz]', labelpad=20, fontsize=16)

    plt.legend()

    plt.savefig(outpath  / ('fwtp_rates_'+ dp.stem + '.pdf'))

def hist_plot(fwtp_df, label_dict, run, tr_num, pdf=True):


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2,5, figsize=(20, 8)); axs = axs.ravel()
    fig.suptitle("run number: %.0f, trigger record : %.0f" %(run, tr_num), y = 0.99)

    c = iter(plt.cm.jet(np.linspace(0, 1,len(header_labels))))

    for n, var in enumerate(header_labels.keys()):
        fwtp_df[var].plot( kind = "hist", range = [min(fwtp_df[var]), max(fwtp_df[var])],
                           bins = 50, histtype = 'stepfilled', ax = axs[n],
                           color = next(c), edgecolor = 'k', alpha =0.8)
        axs[n].set_xlabel(var)
        axs[n].set_axisbelow(True)
        #axs[n].set_yscale('log')
       
    plt.tight_layout()

    if pdf:
        pdf.savefig()

    plt.show()
    plt.close()

#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--input_type', type=click.Choice(["TR", "DF"]),
              help="Select input file type", default='TR', show_default=True)
@click.option('-n', '--tr-num',
              help="Enter trigger numbers to plot, either a single value, a comma-separated list, a colon-separated range or a combination of these")
@click.option('-i', '--interactive', is_flag=True,
              help="Run interactive mode", default=False, show_default=True)
@click.option('-f', '--frame_type', type=click.Choice(["ProtoWIB", "WIB"]),
              help="Select input frame type", default='WIB', show_default=True)
@click.option('-m', '--channel_map_name', type=click.Choice(
    [
        "VDColdboxChannelMap",
        "HDColdboxChannelMap",
        "ProtoDUNESP1ChannelMap",
        "PD2HDChannelMap",
        "VSTChannelMap"
    ]),
    help="Select input channel map", default="HDColdboxChannelMap", show_default=True)
@click.option('--hardware_map_name', type=click.Choice(
    [
        "APA1",
        "APA2"
    ]),
    help="Select input hardware channel map", default="APA1", show_default=True)
@click.option('-t', '--threshold', type=int,
              help="Enter threshold used in run", default=100, show_default=True)
@click.option('-w', '--num-waves', type=int,
              help="Number of 1D waveforms to plot", default=10, show_default=True)
@click.option('-s', '--step', type=int,
              help="Number of TPs to skip when doing 1D plots", default=150, show_default=True)
@click.option('-o', '--outpath', help="Output path for plots", default=".", show_default=True)
@click.option('--log_level', type=click.Choice(
    [
        "DEBUG",
        "INFO",
        "CRITICAL"
    ]), help="Select log level to output", default="INFO", show_default=True)
@click.option('--log_out', is_flag=True,
              help="Redirect log info to file", default=False, show_default=True)
def cli(file_path: str, input_type: str, tr_num, interactive: bool, frame_type: str, channel_map_name: str, hardware_map_name: str, threshold: int, num_waves: int, step: int, outpath: str, log_level: str, log_out: bool) -> None:

    script = Path(__file__).stem
    if log_out:
        logging.basicConfig(
            filename=f'log_{script}_{time.strftime("%Y%m%d")}_{time.strftime("%H%M%S")}.txt',
            filemode="w",
            level=log_level,
            format="%(message)s",
            datefmt="[%X]"
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )

    dp = Path(file_path)
    tr_flag = False

    tr_list = list(tr_num.split(','))
    tr_num = []
    for tr in tr_list:
        if ":" in tr:
            tr_first, tr_last = map(int, tr.split(':'))
            tr_num.extend([*range(tr_first, tr_last+1)])
        else:
            tr_num.append(int(tr))

    rich.print(f'Triggers to extract: {tr_num}')

    if input_type == "TR":
        tr_flag = True
        rdm = DataManager(dp.parent, frame_type, channel_map_name)
        data_files = sorted(rdm.list_files(), reverse=True)
        rich.print(data_files)
        f = dp.name
        rich.print(f)
        trl = rdm.get_entry_list(f)
        rich.print(trl)
        tr_load = trl if tr_num[0] == -1 else tr_num

        entries = []
        for tr in tr_load:
            if tr not in trl:
                raise IndexError(f"{tr} does not exists!")
            try:
                entries.append(rdm.load_entry(file_path, tr))
            except:
                rich.print(f"Error when trying to open record {tr}!")
                pass
        en_info, tpc_df, tp_df, fwtp_df = map(pd.concat, zip(*entries))

        #Throw exception if no FTPs in TR.
        if fwtp_df.empty:
           raise Exception("No TPs found in the trigger record! Terminating script...")
        fwtp_df = fwtp_df.astype({'trigger_number': int})

    elif input_type == "DF":
        key_list = get_key_list(file_path)

        en_info = pd.read_hdf(file_path, key="info")
        tpc_df  = pd.read_hdf(file_path, key="raw_adcs")
        fwtp_df = pd.read_hdf(file_path, key="raw_fwtps")
        if "tps" in key_list:
            tr_flag = True
            tp_df = pd.read_hdf(file_path, key="tps")

    run = en_info.iloc[0].run_number

    rich.print(en_info)
    rich.print(tpc_df)
    rich.print(fwtp_df)
    if tr_flag: rich.print(tp_df)

    outpath = Path(outpath)

    #Add link no to FWTP dataframe
    # NOTE: this shouldn't be here, will move to datamanager eventually
    hw_map = open_hw_map(hardware_map_name)
    rich.print(hw_map)
    fwtp_df['link_no'] = fwtp_df.apply(get_link, hw_map=hw_map, axis=1)
    rich.print(fwtp_df)

    fwtp_rates_plot(fwtp_df, dp, outpath)

    #Select bad TPs based on crate, slot, fiber and tstamp info
    fwtp_bad_link = fwtp_df.loc[fwtp_df["link_no"] == -1]
    fwtp_large_ts = fwtp_df.loc[fwtp_df["ts"] > 9e17]
    fwtp_small_ts = fwtp_df.loc[fwtp_df["ts"] < 1e17]

    fwtp_bad = pd.concat([fwtp_bad_link, fwtp_large_ts, fwtp_small_ts]).drop_duplicates()

    #Plot the FWTP histograms
    fwtp_df = fwtp_df.merge(fwtp_bad, how='left', indicator=True)

    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.dpi'] = 75

    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('fwtp_1d_hists_'+ dp.stem + '.pdf'))
    hist_plot(fwtp_df[fwtp_df['_merge'] == 'left_only'], header_labels, run, tr_num[0], pdf = pdf)    
    pdf.close()

    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('fwtp_1d_hists_bad_'+ dp.stem + '.pdf'))
    hist_plot(fwtp_bad, header_labels, run, tr_num[0], pdf = pdf)    
    pdf.close()

    if interactive:
        import IPython
        IPython.embed(colors="neutral")
    
if __name__ == "__main__":

    cli()
