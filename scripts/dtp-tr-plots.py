#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
import sys
import rich
import logging
import click
from rich import print
from pathlib import Path

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

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.option('-f', '--frame_type', type=click.Choice(
    [
        "ProtoWIB",
        "WIB"
    ]),
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
@click.option('-o', '--out_path', help="Output path for plots", default="./tp.pdf")

def cli(file_path: str, interactive: bool, frame_type: str = 'WIB', map_id: str = 'HDColdbox', out_path: str = "./tp.pdf") -> None:

    dp = Path(file_path)
    print(dp.parent)
    print(dp.name)


    rdm = DataManager(dp.parent, frame_type, map_id)
    data_files = sorted(rdm.list_files(), reverse=True)
    rich.print(data_files)
    f = dp.name
    rich.print(f)
    trl = rdm.get_entry_list(f)
    rich.print(trl)

    pdf = matplotlib.backends.backend_pdf.PdfPages(out_path)
    for i in range(len(trl)):

        rich.print(f"Reading entry {trl[i]}")
        info, tpc_df, tp_df = rdm.load_entry(f, trl[i])
        rich.print(len(tp_df))
        #rich.print(info)
        if(len(tpc_df) == 0): continue
        #rich.print(tp_df)

        for j in range(len(tp_df)):
            time_start = tp_df["time_start"][j]
            time_over_threshold = tp_df["time_over_threshold"][j]
            time_end = tp_df["time_start"][j]+tp_df["time_over_threshold"][j]
            time_peak = tp_df["time_peak"][j]
            channel = tp_df["channel"][j]
            adc_peak = tp_df["adc_peak"][j]

            if(adc_peak < 120): continue
            if(time_end+3072 > tpc_df.index[-1]): continue
            adc_data = tpc_df.loc[find_nearest(tpc_df.index, time_start-3072):find_nearest(tpc_df.index, time_end+3072), channel]
            adc = adc_data.values
            time = adc_data.index.astype(int) - time_peak

            #rms_adc = rms(adc)
            mu = tpc_df[channel].mean()
            median = tpc_df[channel].median()
            sigma = tpc_df[channel].std()
        
            mono = {'family' : 'monospace'}

            wave_info = '\n'.join((
                f'{"mean = ":<18}{round(mu,2):>3}',
                f'{"std = ":<20}{round(sigma,2):>3}',
                f'{"peak adc = ":<22}{adc_peak:>3}',
                f'{"time over threshold = ":<22}{time_over_threshold:>3}'))
        
            record_info = '\n'.join((
                f'{"run number = ":<17}{info["run_number"]:>10}',
                f'{"trigger number = ":<17}{info["trigger_number"]:>10}',
                f'{"channel = ":<17}{channel:>10}',
                f'{"tstamp = ":<9}{time_peak:>10}'))

            fig = plt.figure()
            plt.style.use('ggplot')
            ax = fig.add_subplot(111)
            plt.plot(time, adc, c="dodgerblue", label="Raw ADC")
            #plt.text(0.5, 0.9, str(time[np.argmax(adc)]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            props = dict(boxstyle='square', facecolor='white', alpha=0.8)
            ax.text(0.035, 0.95, wave_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, fontdict=mono)
            ax.text(0.60, 0.175, record_info, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props, fontdict=mono)
            ax.axvspan(time_start-time_peak, time_end-time_peak, alpha=0.2, color='red')
            ax.axvspan((time_start-time_peak)-fir_shift*32, (time_end-time_peak)-fir_shift*32, alpha=0.4, color='red')
            plt.axvline(x=0, linestyle="--", c="k", alpha=0.6)
            plt.axvline(x=-fir_shift*32, linestyle="--", c="k", alpha=0.6)
            plt.axhline(y=median, linestyle="-.", c="k", alpha=0.5, label="median")
            plt.axhline(y=median+100, linestyle="-.", c="limegreen", alpha=0.5, label="median+threshold")
            loc = ticker.MultipleLocator(base=1000)
            ax.xaxis.set_major_locator(loc)
            plt.ylim(np.min(adc)*0.8, np.max(adc)*1.2)
            plt.xlabel("Relative time [ticks]", fontsize=12, labelpad=10, loc="right")
            legend = plt.legend(fontsize=8, loc="upper right")
            frame = legend.get_frame()
            frame.set_color('white')
            frame.set_alpha(0.8)
            frame.set_linewidth(0)
            pdf.savefig()
            plt.close()
    pdf.close()
    
    if interactive:
        import IPython
        IPython.embed()
    
if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(
#        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    cli()
