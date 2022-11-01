#!/usr/bin/env python

from dtpfeedbacktools.datamanager import DataManager
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

def save_hdf5(en_info, adc_df, tp_df, fwtp_df, out_base_name):
    store = pd.HDFStore(out_base_name.with_suffix(".hdf5"))
    print("Saving run info dataframe")
    en_info.to_hdf(store, 'info')
    print("Saving raw tps dataframe")
    fwtp_df.to_hdf(store, 'raw_fwtps')
    print("Saving adcs dataframe")
    adc_df.to_hdf(store, 'raw_adcs')
    print("Saving tp dataframe")
    tp_df.to_hdf(store, 'tps')
    store.close()

def save_csv(en_info, adc_df, tp_df, fwtp_df, out_base_name):
    parent, name = out_base_name.parent, out_base_name.name
    print("Saving run info dataframe")
    en_info_name = name+"_info.csv"
    en_info.to_csv(Path(parent).joinpath(en_info_name))
    print("Saving raw tps dataframe")
    fwtp_df_name = name+"_raw_fwtps.csv"
    fwtp_df.to_csv(Path(parent).joinpath(fwtp_df_name))
    print("Saving adcs dataframe")
    adc_df_name = name+"_raw_adcs.csv"
    adc_df.to_csv(Path(parent).joinpath(adc_df_name))
    print("Saving tp dataframe")
    tp_df_name = name+"_tps.csv"
    tp_df.to_csv(Path(parent).joinpath(tp_df_name))

out_method = {"HDF5": save_hdf5, "CSV": save_csv}

#------------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('file_path', type=click.Path(exists=True))
@click.option('-n', '--tr-num', type=int,
              help="Enter trigger number to export", default=1, show_default=True)
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
@click.option('--out_format', type=click.Choice(["HDF5", "CSV"]),
              help="Select format of output", default='HDF5', show_default=True)
@click.option('-o', '--out_path', help="Output path for plots", default=".", show_default=True)

def cli(file_path: str, tr_num : int, interactive: bool, frame_type: str, map_id: str, out_format: str, out_path: str) -> None:

    dp = Path(file_path)
    out_path = Path(out_path)

    rdm = DataManager(dp.parent, frame_type, map_id)
    data_files = sorted(rdm.list_files(), reverse=True)
    rich.print(data_files)
    f = dp.name
    rich.print(f)
    trl = rdm.get_entry_list(f)
    # rich.print(trl)
    if tr_num not in trl:
        raise IndexError(f"{tr_num} does not exists!");
    en_info, adc_df, tp_df, fwtp_df = rdm.load_entry(file_path, tr_num)

    out_base_name = out_path / (dp.stem + f'_tr_{tr_num}')
    out_method[out_format](en_info, adc_df, tp_df, fwtp_df, out_base_name)
    
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
