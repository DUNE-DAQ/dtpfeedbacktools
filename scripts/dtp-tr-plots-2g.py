#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
import sys
import rich
import logging
import click
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
    # if(adc_peak < 120): continue

    mu = raw_adcs[channel].mean()
    median = raw_adcs[channel].median()
    sigma = raw_adcs[channel].std()

    ts_min = tstamp-pkt_len_ts*n_packets
    ts_max = tstamp+pkt_len_ts*(n_packets+1)
    tp_data = rtp_df[(rtp_df['ts']>ts_min) & (rtp_df['ts']<=ts_max) & (rtp_df['offline_ch']==rtp['offline_ch'])]
    tp_data = tp_data.copy()
    tp_data['ts'] = tp_data['ts']-tstamp
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
    
    # plt.style.use('ggplot')
    plt.style.use('seaborn-v0_8-whitegrid')
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
    plt.plot(time, adc, 'x-', c="powderblue", label="Raw ADC", linewidth=1.5)
    plt.plot(time_del, adc, 'x-', c="dodgerblue", label="Raw ADC + FIR delay", linewidth=1.5)
    
    for i in range(2*n_packets+2):
        plt.axvline(x=-n_packets*pkt_len_ts+i*pkt_len_ts, linestyle="--", c="k", alpha=0.2)
    
    # 
    plt.axvspan(time_start*tick_per_sample, time_end*tick_per_sample, alpha=0.3, color='red')
    plt.axvline(x=time_peak*tick_per_sample, linestyle="-", c="k", alpha=0.3)
    
    ax.hlines(y=fw_median, xmin=0, xmax=2048, linestyle="-.", colors="black", alpha=0.5, label="median")
    ax.hlines(y=fw_median+threshold*fir_correction, xmin=0, xmax=2048, linestyle="-.", colors="limegreen", alpha=0.5, label="median+threshold")
    
    ax.text(0.02, 0.98, wave_info, transform=ax.transAxes, fontsize=8, va='top', bbox=props_wave, fontdict=mono)
    ax.text(0.02, 0.02, tp_info, transform=ax.transAxes, fontsize=8, va='bottom', bbox=props_tp, fontdict=mono)
    ax.text(0.98, 0.02, record_info, transform=ax.transAxes, fontsize=8, ha='right', va='bottom', bbox=props_record, fontdict=mono)
    
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
def plotme_an_ED(df_adc, df_tp, run, ntsamples, zeroped, pdf = None):

    fir_delay = 16
    cmap = cm.get_cmap('bwr',2) #cmap for hit_continue param. 
    norm = colors.LogNorm() # cmap for raw adc data
    
    #Prepare data for plotting 
    chan = pd.to_numeric(df_adc.columns[1:]) 
    df_adc = df_adc.head(ntsamples) # only plot user-specified number of samples 
    Z = df_adc.to_numpy()[:,1:]

    #quick cheated pedsub
    if zeroped:
        Z = Z - np.mean(Z, axis = 0)
        norm = colors.TwoSlopeNorm(vmin=np.min(Z), vcenter=0, vmax=np.max(Z)) #update cmap so it's centered at 0.


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

    # print( min(df_adc.index),max(df_adc.index), min(chan), max(chan) )
    plt.ylabel('offline channel number')
    plt.xlabel('timestamp [tick]')
    plt.legend(title = "run number: %.0f" %run)
    cb = plt.colorbar(ticks = [0.25,0.75], shrink =  0.7)
    cb.ax.set_yticklabels(['0','1'])
    cb.set_label("hit_continue", rotation = 270)
    
    if pdf: pdf.savefig()
    plt.show()
    plt.close()


#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('-n', '--tr-num', type=int, default=1)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('-f', '--frame_type', type=click.Choice(["ProtoWIB", "WIB"]),
              help="Select input frame type", default='WIB')
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default='HDColdbox')
@click.option('-t', '--threshold', type=int, default=100)
@click.option('-w', '--num-waves', type=int, default=10)
@click.option('-s', '--step', type=int, default=150)
@click.option('-o', '--outpath', help="Output path for plots", default=".")

def cli(file_path: str, tr_num : int, interactive: bool, frame_type: str, map_id: str, threshold: int, outpath: str, num_waves: int, step: int) -> None:

    dp = Path(file_path)

    rdm = DataManager(dp.parent, frame_type, map_id)
    data_files = sorted(rdm.list_files(), reverse=True)
    rich.print(data_files)
    f = dp.name
    rich.print(f)
    trl = rdm.get_entry_list(f)
    # rich.print(trl)
    if tr_num not in trl:
        raise IndexError(f"{tr_num} does not exists!");
    en_info, tpc_df, tp_df, fwtp_df = rdm.load_entry(file_path, trl[0])
    
    run = en_info['run_number']

    outpath = Path(outpath)
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('TRDisplay_fwtp_'+ dp.stem + '.pdf'))
    plotme_an_ED( tpc_df, fwtp_df, run, len(tpc_df), True, pdf = pdf)
    pdf.close()

    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('TRDisplay_tp_'+ dp.stem + '.pdf'))
    plotme_an_ED( tpc_df, tp_df, run, len(tpc_df), True, pdf = pdf)
    pdf.close()
    # fwtp_df_centered = fwtp_df[(fwtp_df['hit_continue'] == 0) & (fwtp_df['start_time'] != 0) & (fwtp_df['end_time'] != 63)]

    # outpath = Path(outpath)

    # plt.rcParams['figure.figsize'] = [12., 5.]
    # plt.rcParams['figure.dpi'] = 75
    # pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_centered_waveformas'+ dp.stem + '.pdf'))

    # # 100 and 150 are kind of random pocks to sample the input file
    # for k in range(num_waves):
    #     idx = step*k
    #     rich.print(f"Plotting centered tp  {idx}")
    #     if idx > len(fwtp_df_centered.index):
    #         break
    #     plotme_a_fwtp(fwtp_df_centered.iloc[idx], fwtp_df, tpc_df, idx, run, threshold, 1, pdf=pdf)
  
    # pdf.close()
    
    # pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_edge_waveformas'+ dp.stem + '.pdf'))
    # fwtp_df_edges = fwtp_df[(fwtp_df['hit_continue'] == 1) | (fwtp_df['start_time'] == 0) | (fwtp_df['end_time'] == 63)]
    # for k in range(num_waves):
    #     idx = step*k
    #     rich.print(f"Plotting edge tp  {idx}")
    #     if idx > len(fwtp_df_edges.index):
    #         break

    #     plotme_a_fwtp(fwtp_df_edges.iloc[idx], fwtp_df, tpc_df, idx, run, threshold, 1, pdf=pdf)
    # pdf.close()



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
