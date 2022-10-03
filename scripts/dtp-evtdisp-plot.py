#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from pylab import cm
import matplotlib.backends.backend_pdf
import numpy as np
from pathlib import Path 

import click


plt.rcParams['figure.figsize'] = [18., 10.]
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({'font.size': 12})
plt.style.use("bmh") 

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
    plt.imshow(Z.T, cmap = 'coolwarm',aspect = 'auto', origin = 'lower', norm = norm,
               extent = [ min(df_adc.index),max(df_adc.index), min(chan), max(chan) ] )
    # Overlay the FW hits 
    plt.scatter(df_tp['peak_time'] - fir_delay + df_tp['ts'], df_tp['offline_ch'],  c = df_tp['hit_continue'],
                s = 16, label = 'firmware hits', alpha  =0.9, cmap = cmap)
    print( min(df_adc.index),max(df_adc.index), min(chan), max(chan) )
    plt.ylabel('offline channel number')
    plt.xlabel('timestamp [tick]')
    plt.legend(title = "run number: %.0f" %run)
    cb = plt.colorbar(ticks = [0.25,0.75], shrink =  0.7)
    cb.ax.set_yticklabels(['0','1'])
    cb.set_label("hit_continue", rotation = 270)
    
    if pdf: pdf.savefig()
    plt.show()
    plt.close()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('rawdata_file', type=click.Path(exists=True))
@click.option('-r', '--run', type=int, default=1)
@click.option('-n', '--ntsamples', type=int, default=5000)
@click.option('-o', '--outpath', type=click.Path(file_okay=False), default='.')

def cli(rawdata_file, run, ntsamples, outpath):

    rawdata_file = Path(rawdata_file)
    df_adc = pd.read_hdf(rawdata_file, key="/raw_adcs")
    df_tp = pd.read_hdf(rawdata_file, key="/raw_fwtps")

    outpath = Path(outpath)
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath  / ('EventDisplay_'+ rawdata_file.stem + '.pdf'))
    plotme_an_ED( df_adc, df_tp, run, ntsamples, True, pdf = pdf)
    pdf.close()


if __name__ == '__main__':
    cli()
