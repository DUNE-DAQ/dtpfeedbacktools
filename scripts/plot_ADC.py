import pandas as pd
import numpy as np
import detchannelmaps
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse

plt.rcParams.update({'font.size': 12})
plt.style.use("bmh") 


def evd(df, name : str):
    norm = colors.LogNorm() # cmap for raw adc data

    #Prepare data for plotting 
    chan = pd.to_numeric(df.columns[1:]) 
    Z = df.to_numpy()[:,1:]

    #quick cheated pedsub
    Z = Z - np.mean(Z, axis = 0)


    #2D plot of the raw ADC data
    plt.imshow(Z.T, cmap = 'coolwarm',aspect = 'auto', origin = 'lower', norm = norm,
                extent = [ min(df.index),max(df.index), min(chan), max(chan) ] )
    plt.colorbar()
    plt.savefig(f"event_display_{name}.png", dpi=300)


def wire(df, name : str):
    single_wire = df.iloc[:, 2500]
    print(single_wire)

    plt.plot(single_wire.index, single_wire.to_numpy(), marker="x")
    plt.savefig(f"single_wire_{name}.png", dpi=300)

def main(args):

    filename = args.file
    name = filename.split(".")[0]
    print(f"looking at {name}")
    df = pd.read_hdf(filename, key = "raw_adcs")
    print(df)

    evd(df, name)
    plt.close()
    wire(df, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some raw ADC stuff")
    parser.add_argument(dest="file", type=str, help="file to open.")
    main(parser.parse_args())