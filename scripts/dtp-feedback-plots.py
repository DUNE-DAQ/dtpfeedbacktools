#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
import detchannelmaps
import sys
import warnings
import rich
import logging
import time
import click
import h5py
from rich import print
from rich.logging import RichHandler
from pathlib import Path
from pylab import cm
from matplotlib import colors

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf

fir_shift = 15

def get_key_list(file):
    with h5py.File(file, "r") as f:
        key_list = list(f.keys())
    return key_list

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def rms(array):
    return np.sqrt(np.sum(np.power(array.astype(int), 2))/len(array.astype(int)))

def open_hw_map(hw_map_name):
    hw_map_path = hw_map_name
    hw_map_df = pd.read_csv(hw_map_path, index_col=False, header=1, delimiter=" ", names=["DRO_SourceID", "DetLink", "DetSlot", "DetCrate", "DetID", "DRO_Host", "DRO_Card", "DRO_SLR", "DRO_Link"])
    hw_map = {}
    for i, line in hw_map_df.iterrows():
        hw_map[(line.DetCrate, line.DetSlot, line.DetLink)] = line.DRO_Link+6*line.DRO_SLR
    return hw_map

#------------------------------------------------------------------------------
def plotme_a_fwtp(rtp, rtp_df, raw_adcs, i, run, threshold, fir_correction, pdf=None):
    
    tick_per_sample = 32
    fir_delay = 16
    n_packets = 1
    dy_min = -100
    dy_max = +100
    pkt_len_ts = 32*64
    
    tstamp = rtp["ts"]
    if(i == 0):
        rich.print(tstamp)
    channel = rtp["offline_ch"]
    time_start = rtp["start_time"]
    time_end = rtp["end_time"]
    time_peak = rtp["peak_time"]
    time_over_threshold = time_end - time_start
    adc_peak = rtp["peak_adc"]
    fw_median = rtp["median"]
    accumulator = rtp["accumulator"]
    tp_number = rtp.name
    trigger_number = rtp["trigger_number"]
    # if(adc_peak < 120): continue

    mu = raw_adcs[channel].mean()
    median = raw_adcs[channel].median()
    sigma = raw_adcs[channel].std()

    ts_min = tstamp-pkt_len_ts*n_packets
    ts_max = tstamp+pkt_len_ts*(n_packets+1)
    tp_data = rtp_df[(rtp_df['ts']>ts_min) & (rtp_df['ts']<=ts_max) & (rtp_df['offline_ch']==rtp['offline_ch'])]
    tp_data = tp_data.copy()
    tp_data['ts'] = tp_data['ts']-tstamp
    n_tps = tp_data.shape[0]
    adc_data = raw_adcs.loc[ts_min:ts_max, channel]
    adc = adc_data.values
    time = adc_data.index.astype(int) - tstamp
    time_del = adc_data.index.astype(int) - tstamp + fir_delay*tick_per_sample

    try:
        y_min = min(adc) + dy_min
        y_max = max(adc) + dy_max
    except:
        rich.print("Something went wrong with the ADCs in this window!")
        y_min = fw_median + dy_min
        y_max = fw_median + dy_max

    wave_info = '\n'.join((
        f'{"mean = ":<7}{round(mu,2):>6}',
        f'{"std = ":<7}{round(sigma,2):>6}')
        )

    tp_info = '\n'.join((
        f'{"median = ":<14}{fw_median:>4}',
        f'{"accumulator = ":<14}{accumulator:>4}',
        f'{"peak adc = ":<14}{adc_peak:>4}',
        f'{"tot [tt] = ":<14}{time_over_threshold:>4}'))

    record_info = '\n'.join((
        f'{"run number = ":<17}{run:>10}',
        f'{"trigger number = ":<17}{trigger_number:>10}',
        f'{"channel = ":<17}{channel:>10}',
        f'{"tstamp = ":<9}{tstamp:>10}'))

    fig = plt.figure()
    gs = fig.add_gridspec(1, 3)
    
    plt.style.use('seaborn-v0_8-white')

    with plt.style.context('default'):
        ax = fig.add_subplot(gs[0,2])
        ax.axis('off')
        t = ax.table(cellText=tp_data.T.values, rowLabels=tp_data.T.index, colLabels=tp_data.T.columns, loc='center', edges='open')
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(0.7, 0.9)
    ax = fig.add_subplot(gs[0,0:2])
    
    props_record = dict(boxstyle='square', facecolor='white', alpha=1)
    props_tp = dict(boxstyle='square', facecolor='lightcoral', alpha=0.75)
    props_wave = dict(boxstyle='square', facecolor='lightskyblue', alpha=1)
    mono = {'family' : 'monospace'}
    #plt.plot(time, adc, 'x-', c="powderblue", label="Raw ADC", linewidth=1.5)
    plt.plot(time_del, adc, 'x-', c="dodgerblue", label="Raw ADC", linewidth=1.5)
    
    for i in range(2*n_packets+2):
        plt.axvline(x=-n_packets*pkt_len_ts+i*pkt_len_ts, linestyle="--", c="k", alpha=0.2)
    
    # 
    plt.axvspan(time_start*tick_per_sample, time_end*tick_per_sample, alpha=0.3, color='red', linewidth=0)
    plt.axvline(x=time_peak*tick_per_sample, linestyle="-", c="k", alpha=0.3)

    ax.hlines(y=fw_median, xmin=0, xmax=2048, linestyle="-.", colors="black", alpha=0.5, label="median")
    ax.hlines(y=fw_median+threshold*fir_correction, xmin=0, xmax=2048, linestyle="-.", colors="limegreen", alpha=0.5, label="median+threshold")
    
    ax.text(0.02, 0.98, wave_info, transform=ax.transAxes, fontsize=8, va='top', bbox=props_wave, fontdict=mono)
    ax.text(0.02, 0.02, tp_info, transform=ax.transAxes, fontsize=8, va='bottom', bbox=props_tp, fontdict=mono)
    ax.text(0.98, 0.02, record_info, transform=ax.transAxes, fontsize=8, ha='right', va='bottom', bbox=props_record, fontdict=mono)
    
    for j in range(n_tps):
        tp_no = tp_data.index[j]
        if(tp_no == tp_number): continue
        plt.axvspan(tp_data["ts"][tp_no]+tp_data["start_time"][tp_no]*tick_per_sample, tp_data["ts"][tp_no]+tp_data["end_time"][tp_no]*tick_per_sample, alpha=0.3, color='lightpink', linewidth=0)
        plt.axvline(x=tp_data["ts"][tp_no]+tp_data["peak_time"][tp_no]*tick_per_sample, linestyle=":", linewidth=1, c="k", alpha=0.2)

    plt.ylim(y_min, y_max)
    
    plt.xlabel("Relative time [ticks]", fontsize=12, labelpad=10, loc="right")
    
    legend = plt.legend(fontsize=8, loc="upper right")
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_alpha(0.8)
    frame.set_linewidth(0)
    if pdf:
        pdf.savefig()
    # fig.tight_layout()

    plt.show()
    plt.close()

#------------------------------------------------------------------------------
def plotme_an_ADC_ED(df_adc, run, planeID, ntsamples, zeroped, pdf = None):
    norm = colors.LogNorm() # cmap for raw adc data

    #Prepare data for plotting 
    df_adc = df_adc.loc[df_adc.index[ntsamples[0]]:df_adc.index[ntsamples[1]]]

    # timestamp for the beginning of data capture that t will be plotted relative to
    relative_ts = df_adc.index - df_adc.index[0]

    Z = df_adc.to_numpy()[:,1:]

    #quick cheated pedsub
    if zeroped:
        Z = Z - np.median(Z, axis = 0)
        rms = np.mean(np.sqrt(np.mean(Z**2, axis = 0)))
        #update cmap so it's centered at 0.
        norm = colors.TwoSlopeNorm(vmin = -5 * rms, vcenter = 0, vmax = 5 * rms)


    #2D plot of the raw ADC data
    im = plt.imshow(Z.T, cmap = 'RdBu_r',aspect = 'auto', origin = 'lower', norm = norm,
            extent = [ min(relative_ts),max(relative_ts), min(df_adc.columns), max(df_adc.columns) ] )

    cb1 = plt.colorbar(im, shrink = 0.7)
    cb1.set_label("ADC ", rotation = 270, labelpad = +20)
    plt.xlabel("relative time [tick]")
    plt.ylabel("offline channel")
    plt.title(f"PlaneID: {planeID} | start timestamp: {df_adc.index[0]}")
    plt.legend(title = f"run number: {run}")
    plt.tight_layout()

    if pdf: pdf.savefig()
    plt.show()
    plt.close()


#------------------------------------------------------------------------------
def plotme_an_ED_v2(df_adc, df_tp, run, ntsamples, zeroped, pdf = None):

    # Detect if it's a fwtp or tp dataframe
    is_fw_tp = 'ts' in df_tp.columns

    fir_delay = 16
    ticks_per_sample = 32

    cmap = cm.get_cmap('copper',2) #cmap for hit_continue param. 
    norm = colors.LogNorm() # cmap for raw adc data
    # timestamp for the beginning of data capture that t will be plotted relative to
    t0 = df_adc.index[0]
    
    #Prepare data for plotting 
    df_adc = df_adc.head(ntsamples) # only plot user-specified number of samples 
    Z = df_adc.to_numpy()[:,1:]

    #quick cheated pedsub
    if zeroped:
        Z = Z - np.mean(Z, axis = 0)
        #update cmap so it's centered at 0.
        norm = colors.TwoSlopeNorm(vmin=np.min(Z), vcenter=0, vmax=np.max(Z))


    #convert the ylabels to account for channels not being consecutive     
    y_vals = df_adc.columns.to_list()[0:]; y_vals = [int(i) for i in y_vals]

    y=df_tp['offline_ch'].map(lambda k: y_vals.index(k))

    n_yticks = 8 #number of ticks to plot of y-axis regardless of n_channels
    yloc  = [i for i in range(1, len(df_adc.columns), int(len(df_adc.columns)/n_yticks))]   
    yticks = [str(df_adc.columns[i]) for i in yloc]

    #Plot the data
    fig, (ax1, ax2) = plt.subplots( 2, sharex= True, figsize = (15,10))

    for i in [ax1, ax2]:
        im = i.imshow(Z.T, cmap = 'RdBu_r',aspect = 'auto', origin = 'lower', norm = norm)
        i.set_ylabel('offline channel number')
        i.set_xlim(0, len(Z))
        i.set_yticks(yloc, yticks )
        i.set_ylim(yloc[0],(yloc[-1]))
    
    #Overlay the 2d hits
    # hits = ax2.scatter((df_tp['peak_time'] - fir_delay + (df_tp['ts'] -t0)/ticks_per_sample), y.values,
    #                    c = df_tp['hit_continue'], s = 15, label = 'firmware hits', alpha =0.6, cmap = cmap)
    if is_fw_tp:
        hits = ax2.scatter(
            (df_tp['peak_time'] - fir_delay + (df_tp['ts'] -t0)/ticks_per_sample), 
            y.values,  
            c = df_tp['hit_continue'],
            s = 15,
            marker = ",",
            label = 'firmware hits', 
            alpha  =0.6, 
            cmap = cmap
        )
    else:
        hits = ax2.scatter((df_tp['peak_time']-t0)/ticks_per_sample, y.values, s = 15, label = 'stitched hits', alpha =0.6)

                       
    cb1 = plt.colorbar(im, ax = ax1, shrink = 0.7)
    cb1.set_label("ADC ", rotation = 270, labelpad = +20)
    if is_fw_tp:
        cb2 = plt.colorbar(hits, ax = ax2, ticks = [0.25,0.75], shrink =  0.7)
        cb2.ax.set_yticklabels(['0','1'])
        cb2.set_label('hit_continue ', rotation = 270, labelpad = +20)
    ax2.set_xlabel('relative time [tick]')
    plt.legend(title = "run number: %.0f" %run)
    plt.tight_layout()

    if pdf: pdf.savefig()
    plt.show()
    plt.close()


def plotme_a_channel(tpc_df : pd.DataFrame, run : int, channel : int = 0, pdf : matplotlib.backends.backend_pdf.PdfPages = None):
    """ Plots the ADC data of specified channels.

    Args:
        tpc_df (pd.DataFrame): ADC data
        run (int) run number 
        channel (int, optional): channel to plot. Defaults to 0.
        pdf (matplotlib.backends.backend_pdf.PdfPages, optional): pdf backend to use. Defaults to None.
    """
    # timestamp for the beginning of data capture that t will be plotted relative to
    t0 = tpc_df.index[0]

    try:
        single_channel = tpc_df.iloc[:, channel] # get data from our signle channel
        print(f"plotting channel: {channel}")
        
        with plt.style.context('bmh'): # lazy formatting
            plt.plot(single_channel.index - t0, single_channel.to_numpy(), marker = "x", label = f"offline channel: {channel}")
        plt.title(f"first timestamp: {t0}")
        plt.xlabel("relative timestamp [tick]")
        plt.ylabel("ADC")
        plt.legend(title = f"run number: {run}")
        plt.tight_layout()

        if pdf: pdf.savefig()
        plt.show()
        plt.close()
    except:
        warnings.warn(f"channel {channel} was not found in the dataframe.")
        pass


def parse_number_list(numbers : str):
    """ Parse a list of numbers stored in string form e.g. 1,2,3:5 -> [1,2,3,4,5]
    Args:
        numbers (str) : string to parse
    """
    sets = list(numbers.split(',')) # split sets
    numbers_list = []
    for s in sets:
        if ":" in s: # iterate through number in a set if it specifies a range
            first, last = map(int, s.split(':'))
            numbers_list.extend([*range(first, last+1)])
        else:
            numbers_list.append(int(s))
    return numbers_list

#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--hardware_map_file', type=click.Path(exists=True),
              help="Select input hardware channel map")
@click.option('--input_type', type=click.Choice(["TR", "DF"]),
              help="Select input file type", default='TR', show_default=True)
#@click.option('-n', '--tr-num', type=int,
#              help="Enter trigger number to plot", default=1, show_default=True)
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
@click.option('-t', '--threshold', type=int,
              help="Enter threshold used in run", default=100, show_default=True)
@click.option('-w', '--num-waves', type=int,
              help="Number of 1D waveforms to plot", default=10, show_default=True)
@click.option('-s', '--step', type=int,
              help="Number of TPs to skip when doing 1D plots", default=150, show_default=True)
@click.option('-c', '--channel', type=str,
              help="offline channel to plot, either a single value, a comma-separated list, a colon-separated range or a combination of these", default=0, show_default=True)
@click.option('--time_range', type=str, default = "0:1", help="fractional time range for ADC event display, colon separated for min max range e.g. min:max")
@click.option('-o', '--outpath', help="Output path for plots", default=".", show_default=True)
@click.option('--log_level', type=click.Choice(
    [
        "DEBUG",
        "INFO",
        "CRITICAL"
    ]), help="Select log level to output", default="INFO", show_default=True)
@click.option('--log_out', is_flag=True,
              help="Redirect log info to file", default=False, show_default=True)
def cli(file_path: str, hardware_map_file: str, input_type: str, tr_num, interactive: bool, frame_type: str, channel_map_name: str, threshold: int, num_waves: int, step: int, channel : str, time_range : str, outpath: str, log_level: str, log_out: bool) -> None:
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

    print(f"{hardware_map_file=}")
    # open_hw_map(hardware_map_file)
     #return

    dp = Path(file_path)
    tr_flag = False

    tr_list = parse_number_list(tr_num)
    rich.print(f'Triggers to extract: {tr_list}')

    channel = parse_number_list(channel)

    if input_type == "TR":
        tr_flag = True
        rdm = DataManager(dp.parent, frame_type, channel_map_name)
        data_files = sorted(rdm.list_files(), reverse=True)
        rich.print(data_files)
        f = dp.name
        rich.print(f)
        trl = rdm.get_entry_list(f)
        rich.print(trl)
        tr_load = trl if tr_list[0] == -1 else tr_list
        #rich.print(tr_load)

        #en_info, tpc_df, tp_df, fwtp_df = zip(*[rdm.load_entry(file_path, tr) if tr in trl else raise IndexError(f"{tr} does not exists!") for tr in tr_load])
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
        if not fwtp_df.empty:
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
    #rich.print(fwtp_df)[{tr_num}]
    if tr_flag: rich.print(tp_df)

    outpath = Path(outpath)

    if not tpc_df.empty:
        time_range = time_range.split(":", 1)
        time_range = [int(len(tpc_df) * float(time_range[0])), int(len(tpc_df) * float(time_range[1])) - 1]

        pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / (f'TRDisplay_adc_evd[{tr_num}]_{dp.stem}.pdf'))        

        # split the df by channel type for event displays
        cmap = detchannelmaps.make_map(channel_map_name) # remake channel map in case we are looking at exported TRs

        planes, start_index = np.unique([cmap.get_plane_from_offline_channel(i) for i in tpc_df.columns], return_index = True)

        start_index = np.append(start_index, len(tpc_df.columns))
        
        for p in range(len(planes)):
            plotme_an_ADC_ED(tpc_df.iloc[:, start_index[p]:start_index[p + 1]], run, planes[p], time_range, True, pdf)

        pdf.close()

        pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / (f'TRDisplay_adc_channels[{tr_num}]_{dp.stem}.pdf'))
        rich.print(f'ADC Channels to plot: {channel}')
        for c in channel:        
            plotme_a_channel(tpc_df, run, c, pdf)
        pdf.close()

    print(tp_df)

    if tr_flag and not tp_df.empty and tr_list != -1:
        pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / (f'TRDisplay_tp_tr[{tr_num}]_{dp.stem}.pdf'))
        plotme_an_ED_v2(tpc_df, tp_df, run, len(tpc_df), True, pdf = pdf)
        pdf.close()

    if not fwtp_df.empty:
        # 2d event displays
        if tr_list != -1:
            pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / (f'TRDisplay_fwtp_tr[{tr_num}]_{dp.stem}.pdf'))
            plotme_an_ED_v2(tpc_df, fwtp_df, run, len(tpc_df), True, pdf = pdf)
            pdf.close()

        # centred waveforms + tps
        fwtp_df_centered = fwtp_df[(fwtp_df['hit_continue'] == 0) & (fwtp_df['start_time'] != 0) & (fwtp_df['end_time'] != 63)]

        plt.rcParams['figure.figsize'] = [12., 5.]
        plt.rcParams['figure.dpi'] = 75

        pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_centered_waveforms_'+ dp.stem + '.pdf'))
        # 100 and 150 are kind of random pocks to sample the input file
        for k in range(num_waves):
            idx = step*k
            rich.print(f"Plotting centered tp  {idx}")
            if idx >= len(fwtp_df_centered.index):
                break
            plotme_a_fwtp(fwtp_df_centered.iloc[idx], fwtp_df, tpc_df, idx, run, threshold, 1, pdf=pdf)
    
        pdf.close()
        
        # edge waveforms + tps
        pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_edge_waveforms_'+ dp.stem + '.pdf'))
        fwtp_df_edges = fwtp_df[(fwtp_df['hit_continue'] == 1) | (fwtp_df['start_time'] == 0) | (fwtp_df['end_time'] == 63)]
        for k in range(num_waves):
            idx = step*k
            rich.print(f"Plotting edge tp  {idx}")
            if idx >= len(fwtp_df_edges.index):
                break

            plotme_a_fwtp(fwtp_df_edges.iloc[idx], fwtp_df, tpc_df, idx, run, threshold, 1, pdf=pdf)
        pdf.close()

    if interactive:
        import IPython
        IPython.embed(colors="neutral")
    
if __name__ == "__main__":

    cli()
