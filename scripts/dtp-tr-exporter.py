#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
import rich
import logging
import time
import click
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


def save_hdf5(en_info, adc_df, tp_df, fwtp_df, out_base_name, tr_num):
    store = pd.HDFStore(out_base_name.with_suffix(".hdf5"))
    head = f"record_{tr_num}/"
    print(f"Saving record {tr_num}")
    en_info.to_hdf(store, head + 'info')
    fwtp_df.to_hdf(store, head + 'raw_fwtps')
    adc_df.to_hdf(store, head + 'raw_adcs')
    tp_df.to_hdf(store, head + 'tps')
    
    store.close()


def save_csv(en_info, adc_df, tp_df, fwtp_df, out_base_name, tr_num):
    parent, name = out_base_name.parent, out_base_name.name
    head = f"_record_{tr_num}"
    en_info_name = name + head + "_info.csv"
    en_info.to_csv(Path(parent).joinpath(en_info_name))
    fwtp_df_name = name + head + "_raw_fwtps.csv"
    fwtp_df.to_csv(Path(parent).joinpath(fwtp_df_name))
    adc_df_name = name + head + "_raw_adcs.csv"
    adc_df.to_csv(Path(parent).joinpath(adc_df_name))
    tp_df_name = name + head + "_tps.csv"
    tp_df.to_csv(Path(parent).joinpath(tp_df_name))

out_method = {"HDF5": save_hdf5, "CSV": save_csv}

#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
#@click.option('-n', '--tr-num', type=int,
#              help="Enter trigger number to export", default=1, show_default=True)
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
@click.option('--out_format', type=click.Choice(["HDF5", "CSV"]),
              help="Select format of output", default='HDF5', show_default=True)
@click.option('-o', '--out_path', help="Output path for plots", default=".", show_default=True)
@click.option('--log_level', type=click.Choice(
    [
        "DEBUG",
        "INFO",
        "CRITICAL"
    ]), help="Select log level to output", default="INFO", show_default=True)
@click.option('--log_out', is_flag=True,
              help="Redirect log info to file", default=False, show_default=True)
def cli(file_path: str, tr_num, interactive: bool, frame_type: str, channel_map_name: str, out_format: str, out_path: str, log_level: str, log_out: bool) -> None:
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
    out_path = Path(out_path)

    if tr_num == "-1":
        tr_str = ""
    else:
        tr_str = tr_num
        tr_str = tr_str.replace(",", "-").replace(":", "_")
        tr_str = f'_tr_{tr_str}'

    tr_list = list(tr_num.split(','))
    tr_num = []
    for tr in tr_list:
        if ":" in tr:
            tr_first, tr_last = map(int, tr.split(':'))
            tr_num.extend([*range(tr_first, tr_last+1)])
        else:
            tr_num.append(int(tr))

    rich.print(f'Triggers to extract: {tr_num}')

    rdm = DataManager(dp.parent, frame_type, channel_map_name)
    data_files = sorted(rdm.list_files(), reverse=True)
    rich.print(data_files)
    f = dp.name
    rich.print(f)
    trl = rdm.get_entry_list(f)
    tr_load = trl if tr_num[0] == -1 else tr_num
    # rich.print(trl)

    entries = []
    for tr in tr_load:
        if tr not in trl:
            raise IndexError(f"{tr} does not exists!")
        try:
            entries.append(rdm.load_entry(file_path, tr))
        except:
            rich.print(f"Error when trying to open record {tr}!")
            pass

    out_base_name = out_path / (dp.stem + tr_str)
    for i in range(len(entries)):
        out_method[out_format](*entries[i], out_base_name, i)
    
    if interactive:
        import IPython
        IPython.embed(colors="neutral")
    
if __name__ == "__main__":

    cli()
