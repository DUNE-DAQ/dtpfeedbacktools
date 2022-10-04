#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from pylab import cm
import matplotlib.backends.backend_pdf
import numpy as np
from pathlib import Path 

import click


plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({'font.size': 14})
plt.style.use("bmh") 

def plotme_an_ED(df_adc, df_tp, run, ntsamples, zeroped, pdf = None):

    fir_delay = 16
    ticks_per_sample = 32
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


    #convert the ylabels to account for channels not being consecutive     
    y_vals = df_adc.columns.to_list()[0:]; y_vals = [int(i) for i in y_vals]

    y=df_tp['offline_ch'].map(lambda k: y_vals.index(k))

    yloc = [i for i in range(1,len(df_adc.columns),32)]
    yticks = [str(df_adc.columns[i]) for i in yloc]



    #Plot the data
    fig, (ax1, ax2) = plt.subplots( 2, sharex= True, figsize = (15,10))

    for i in [ax1, ax2]:
        im = i.imshow(Z.T, cmap = 'coolwarm',aspect = 'auto', origin = 'lower', norm = norm)
        i.set_ylabel('offline channel number')
        i.set_xlim(0, len(Z))
        i.set_yticks(yloc, yticks )
        i.set_ylim(yloc[0],(yloc[-1]))
    
    #Overlay the 2d hits
    hits = ax2.scatter((df_tp['peak_time'] - fir_delay + (df_tp['ts'] -t0)/ticks_per_sample), y.values,
                       c = df_tp['hit_continue'], s = 15, label = 'firmware hits', alpha  =0.6, cmap = cmap)
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
