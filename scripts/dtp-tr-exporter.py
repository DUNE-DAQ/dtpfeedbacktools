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
@click.option('-o', '--outpath', help="Output path for plots", default=".")

def cli(file_path: str, tr_num : int, interactive: bool, frame_type: str, map_id: str, outpath: str) -> None:

    dp = Path(file_path)
    outpath = Path(outpath)

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

    store = pd.HDFStore( outpath / (dp.stem + f'_tr_{tr_num}.hdf5'))
    print("Saving run info dataframe")
    en_info.to_hdf(store, 'info')
    print("Saving raw tps dataframe")
    fwtp_df.to_hdf(store, 'raw_fwtps')
    print("Saving adcs dataframe")
    adc_df.to_hdf(store, 'raw_adcs')
    print("Saving tp dataframe")
    tp_df.to_hdf(store, 'tps')
    store.close()



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
