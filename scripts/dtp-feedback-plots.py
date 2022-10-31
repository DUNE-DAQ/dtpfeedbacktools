#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
import sys
import rich
import logging
import click
import h5py
from rich import print
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


#------------------------------------------------------------------------------
def plotme_a_fwtp(rtp, rtp_df, raw_adcs, i, run, threshold, fir_correction, pdf=None):
    
    tick_per_sample = 32
    fir_delay = 16
    n_packets = 1
    dy_min = -800
    dy_max = 9000
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

    plt.ylim(median+dy_min, median+dy_max)
    
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
def plotme_an_ED(df_adc, df_tp, run, ntsamples, zeroped, stitch = False, pdf = None):

    rich.print("df_tp", df_tp)

    is_fw_tp = 'peak_time' in df_tp.columns

    fir_delay = 16*32
    cmap = cm.get_cmap('bwr',2) #cmap for hit_continue param. 
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


    #2D plot of the raw ADC data
    plt.imshow(Z.T, cmap = 'RdBu_r',aspect = 'auto', origin = 'lower', norm = norm,
               extent = [ min(df_adc.index),max(df_adc.index), min(chan), max(chan) ] )
    # Overlay the FW hits 

    if 'peak_time' in df_tp.columns:
        plt.scatter(df_tp['peak_time']*32 - fir_delay + df_tp['ts'], df_tp['offline_ch'],  c = df_tp['hit_continue'],
                    s = 16, label = 'firmware hits', alpha  =0.9, cmap = cmap)
    else:
        plt.scatter(df_tp['time_peak'], df_tp['channel'],
                    s = 16, label = 'stiched hits', alpha =0.9)

    hits = ax2.scatter(x.values, y.values,
                       c = df_tp['hit_continue'], s = 100, label = 'firmware hits',
                       alpha = 1, edgecolors=None, linewidths=0, cmap = cmap)
    cb1 = plt.colorbar(im, ax = ax1, shrink = 0.7)
    cb1.set_label("ADC ", rotation = 270, labelpad = +20)
    cb2 = plt.colorbar(hits, ax = ax2, ticks = [0.25,0.75], shrink =  0.7)
    cb2.ax.set_yticklabels(['0','1'])
    cb2.set_label('hit_continue ', rotation = 270, labelpad = +20)
    ax2.set_xlabel('relative time [tick]')
    plt.legend(title = "run number: %.0f" %run)
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

    yloc = [i for i in range(1,len(df_adc.columns),32)]
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


#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--input_type', type=click.Choice(["TR", "DF"]),
              help="Select input file type", default='TR', show_default=True)
@click.option('-n', '--tr-num', type=int,
              help="Enter trigger number to plot", default=1, show_default=True)
@click.option('-i', '--interactive', is_flag=True,
              help="Run interactive mode", default=False, show_default=True)
@click.option('-f', '--frame_type', type=click.Choice(["ProtoWIB", "WIB"]),
              help="Select input frame type", default='WIB', show_default=True)
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default='HDColdbox', show_default=True)
@click.option('-t', '--threshold', type=int,
              help="Enter threshold used in run", default=100, show_default=True)
@click.option('-w', '--num-waves', type=int,
              help="Number of 1D waveforms to plot", default=10, show_default=True)
@click.option('-s', '--step', type=int,
              help="Number of TPs to skip when doing 1D plots", default=150, show_default=True)
@click.option('-o', '--outpath', help="Output path for plots", default=".", show_default=True)

def cli(file_path: str, input_type: str, tr_num : int, interactive: bool, frame_type: str, map_id: str, threshold: int, outpath: str, num_waves: int, step: int) -> None:

    dp = Path(file_path)
    tr_flag = False

    if input_type == "TR":
        tr_flag = True
        rdm = DataManager(dp.parent, frame_type, map_id)
        data_files = sorted(rdm.list_files(), reverse=True)
        rich.print(data_files)
        f = dp.name
        rich.print(f)
        trl = rdm.get_entry_list(f)
        # rich.print(trl)
        if tr_num not in trl:
            raise IndexError(f"{tr_num} does not exists!");
        en_info, tpc_df, tp_df, fwtp_df = rdm.load_entry(file_path, tr_num)
    
    elif input_type == "DF":
        key_list = get_key_list(file_path)

        en_info = pd.read_hdf(file_path, key="info")
        tpc_df  = pd.read_hdf(file_path, key="raw_adcs")
        fwtp_df = pd.read_hdf(file_path, key="raw_fwtps")
        if "tps" in key_list:
            tr_flag = True
            tp_df = pd.read_hdf(file_path, key="tps")

    run = en_info['run_number'][0]

    rich.print(fwtp_df)
    if tr_flag: rich.print(tp_df)

    outpath = Path(outpath)

    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / (f'TRDisplay_fwtp_tr{tr_num}_{dp.stem}.pdf'))
    plotme_an_ED_v2( tpc_df, fwtp_df, run, len(tpc_df), True, pdf = pdf)
    pdf.close()

    if tr_flag:
        pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / (f'TRDisplay_tp_tr{tr_num}_{dp.stem}.pdf'))
        plotme_an_ED_v2( tpc_df, tp_df, run, len(tpc_df), True, pdf = pdf)
        pdf.close()

    fwtp_df_centered = fwtp_df[(fwtp_df['hit_continue'] == 0) & (fwtp_df['start_time'] != 0) & (fwtp_df['end_time'] != 63)]

    plt.rcParams['figure.figsize'] = [12., 5.]
    plt.rcParams['figure.dpi'] = 75

    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_centered_waveformas'+ dp.stem + '.pdf'))
    # 100 and 150 are kind of random pocks to sample the input file
    for k in range(num_waves):
        idx = step*k
        rich.print(f"Plotting centered tp  {idx}")
        if idx > len(fwtp_df_centered.index):
            break
        plotme_a_fwtp(fwtp_df_centered.iloc[idx], fwtp_df, tpc_df, idx, run, threshold, 1, pdf=pdf)
  
    pdf.close()
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_edge_waveformas'+ dp.stem + '.pdf'))
    fwtp_df_edges = fwtp_df[(fwtp_df['hit_continue'] == 1) | (fwtp_df['start_time'] == 0) | (fwtp_df['end_time'] == 63)]
    for k in range(num_waves):
        idx = step*k
        rich.print(f"Plotting edge tp  {idx}")
        if idx > len(fwtp_df_edges.index):
            break

        plotme_a_fwtp(fwtp_df_edges.iloc[idx], fwtp_df, tpc_df, idx, run, threshold, 1, pdf=pdf)
    pdf.close()

    if interactive:
        import IPython
        IPython.embed(colors="neutral")
    
if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(
       level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    cli()