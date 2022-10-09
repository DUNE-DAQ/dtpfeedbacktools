#!/usr/bin/env python

from dtpemulator.tpgmanager import TPGManager

from email.policy import default
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import rich
from pathlib import Path


import click

plt.rcParams['figure.figsize'] = [12., 5.]
plt.rcParams['figure.dpi'] = 75

def plotme_a_fwtp(rtp, rtp_df, raw_adcs, i, run, threshold, fir_correction, emulation, pdf=None):
    
    tick_per_sample = 32
    fir_delay = 16
    n_packets = 1
    dy_min = -800
    dy_max = 9000
    pkt_len_ts = 32*64
    
    # rtp = rtp_df.iloc[i]
    
    tstamp = rtp["ts"]
    # if(i == 0):
    #     rich.print(tstamp)
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
    # rich.print(tp_data)
    n_tps = tp_data.shape[0]
    adc_data = raw_adcs.loc[ts_min:ts_max-tick_per_sample, channel]
    adc = adc_data.values
    time = adc_data.index.astype(int) - tstamp
    time_del = adc_data.index.astype(int) - tstamp + fir_delay*tick_per_sample

    tpgm = TPGManager(1000, "data/fir_coeffs.dat", 6, threshold, "HDColdbox")
    fir_data = raw_adcs.loc[(ts_min-64*tick_per_sample):ts_max-tick_per_sample]
    tp_df, ped_df, pedval_df, fir_df = tpgm.run_channel(fir_data, channel, pedchan=True, ped_debug=True)
    tp_df['ts'] = tp_df['ts']-tstamp

    fir_adc = fir_df.values
    time_fir = fir_df.index.astype(int) - tstamp

    #import IPython
    #IPython.embed()

    tp_join = pd.concat([tp_data, tp_df],join='outer', axis=0, ignore_index=True)
    rich.print(tp_join)
    
    sum_fir = fir_df.loc[fir_df[channel]>threshold].sum()

    wave_info = '\n'.join((
        f'{"mean = ":<7}{round(mu,2):>6}',
        f'{"std = ":<7}{round(sigma,2):>6}')
        )

    fir_info = '\n'.join((
        f'{"sum adc = ":<10}{sum_fir[channel]:>6}',
        ))

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

    # plt.style.use('"cyberpunk"')
    plt.style.use('seaborn-whitegrid')
    
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
    props_fir = dict(boxstyle='square', facecolor='darkred', alpha=0.6)
    props_wave = dict(boxstyle='square', facecolor='dodgerblue', alpha=0.6)
    mono = {'family' : 'monospace'}

    plt.plot(time, adc, 'x-', c="powderblue", label="Raw ADC", linewidth=1.5)
    plt.plot(time_del, adc, 'x-', c="dodgerblue", label="Raw ADC + FIR delay", linewidth=1.5)
    if emulation:
        plt.plot(time_fir, fir_adc + pedval_df.values, 'x-', c="darkred", alpha=0.9, label="Emu FIR ADC", linewidth=1.5)
    
    for i in range(2*n_packets+2):
        plt.axvline(x=-n_packets*pkt_len_ts+i*pkt_len_ts, linestyle="--", c="k", alpha=0.2)
    
    plt.axvspan(time_start*tick_per_sample, time_end*tick_per_sample, alpha=0.3, color='red', linewidth=0)
    # plt.axvspan((time_start-fir_delay)*tick_per_sample, (time_end-15)*tick_per_sample, alpha=0.3, color='red')
    plt.axvline(x=time_peak*tick_per_sample, linestyle="-", c="k", alpha=0.3)
    # plt.axvline(x=(time_peak-fir_delay)*tick_per_sample, linestyle="-", c="k", alpha=0.6)

    for j in range(n_tps - 1):
        tp_no = tp_data.index[j+1]
        plt.axvspan(tp_data["ts"][tp_no]+tp_data["start_time"][tp_no]*tick_per_sample, tp_data["ts"][tp_no]+tp_data["end_time"][tp_no]*tick_per_sample, alpha=0.3, color='red', linewidth=0)
        plt.axvline(x=tp_data["ts"][tp_no]+tp_data["peak_time"][tp_no]*tick_per_sample, linestyle=":", linewidth=1, c="k", alpha=0.2)
    
    ax.hlines(y=fw_median, xmin=0, xmax=2048, linestyle="-.", colors="black", alpha=0.5, label="median")
    ax.hlines(y=fw_median+threshold*fir_correction, xmin=0, xmax=2048, linestyle="-.", colors="limegreen", alpha=0.5, label="median+threshold")
    
    ax.text(0.02, 0.98, wave_info, transform=ax.transAxes, fontsize=8, va='top', bbox=props_wave, fontdict=mono)
    if emulation:
        ax.text(0.02, 0.85, fir_info, transform=ax.transAxes, fontsize=8, va='top', bbox=props_fir, fontdict=mono)
    ax.text(0.02, 0.02, tp_info, transform=ax.transAxes, fontsize=8, va='bottom', bbox=props_tp, fontdict=mono)
    ax.text(0.98, 0.02, record_info, transform=ax.transAxes, fontsize=8, ha='right', va='bottom', bbox=props_record, fontdict=mono)
    
    #plt.xlim(-n_packets*pkt_len_ts, -n_packets*pkt_len_ts+i*pkt_len_ts)
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

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('rawdata_file', type=click.Path(exists=True))
@click.option('-r', '--run', type=int, default=1)
@click.option('-t', '--threshold', type=int, default=100)
@click.option('-e', '--emulation', is_flag=True, default=False)
@click.option('-o', '--outpath', type=click.Path(file_okay=False), default='.')
def cli(rawdata_file, run, threshold, emulation, outpath):
    
    rawdata_file = Path(rawdata_file)
    raw_fwtps = pd.read_hdf(rawdata_file, 'raw_fwtps')
    ts_fwtps_min = int(raw_fwtps["ts"].min())
    raw_adcs = pd.read_hdf(rawdata_file, 'raw_adcs')
    ts_adcs_min = int(raw_adcs.index.min())

    rich.print(raw_adcs)

    raw_fwtps_centered = raw_fwtps[(raw_fwtps['hit_continue'] == 0) & (raw_fwtps['start_time'] != 0) & (raw_fwtps['end_time'] != 63)]

    outpath = Path(outpath)
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_centered_waveforms_'+ rawdata_file.stem + '.pdf'))

    n = 10
    m = 150
    # 100 and 150 are kind of random pocks to sample the input file
    for k in range(n):
        idx = m*k
        rich.print(f"Plotting centered tp  {idx}")
        if idx > len(raw_fwtps_centered.index):
            break
        plotme_a_fwtp(raw_fwtps_centered.iloc[idx], raw_fwtps, raw_adcs, idx, run, threshold, 1, emulation, pdf=pdf)
  
    pdf.close()
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('hit_edge_waveforms_'+ rawdata_file.stem + '.pdf'))
    raw_fwtps_edges = raw_fwtps[(raw_fwtps['hit_continue'] == 1) | (raw_fwtps['start_time'] == 0) | (raw_fwtps['end_time'] == 63)]
    for k in range(n):
        idx = m*k
        rich.print(f"Plotting edge tp  {idx}")
        if idx > len(raw_fwtps_edges.index):
            break

        plotme_a_fwtp(raw_fwtps_edges.iloc[idx], raw_fwtps, raw_adcs, idx, run, threshold, 1, emulation, pdf=pdf)
    pdf.close()

if __name__ == '__main__':
    cli()
