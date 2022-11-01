# dtpfeedbacktools - Tools for DTP prompt feedback

Simple package developed to validate the recorded data and firmware functioning. Data can be visualised and quality inspected almost instantaneously after capture. Some of the features it offers are:

1. Quick measurement of usable overlap in data captures (binary dumps).
2. Detailed comparison for raw data, FW TPs and TPs on individual channels.
3. 2D event displays to check the overall quality of the capture.
4. Possible emulation of the TPG block ([dtpemulator](https://github.com/DUNE-DAQ/dtpemulator)).

## Requirements

The current version of this package relies on `rc-dunedaq-v3.2.0-3`, please check this [link](https://github.com/DUNE-DAQ/daqconf/wiki/Instructions-for-setting-up-a-development-software-area) with the latest instructions on how to set up a development software area.

A set of particular Python packages are required to run these tools. Make sure to include them in your environment by running:
```
pip install -r requirements.txt
```

## Feedback from Trigger Record files

The main inputs for validation are currently Trigger Records (TRs). These contain the raw ADC data, firmware TPs and TPs. After a run is taken and the TRs are written one would like to run `dtp-feedback-plots.py`. To see the different inputs you can provide to it type:
```
dtp-feedback-plots.py --help
```
which should return:
```
Usage: dtp-feedback-plots.py [OPTIONS] FILE_PATH

Options:
  --input_type [TR|DF]            Select input file type  [default: TR]
  -n, --tr-num INTEGER            Enter trigger number to plot  [default: 1]
  -i, --interactive               Run interactive mode  [default: False]
  -f, --frame_type [ProtoWIB|WIB]
                                  Select input frame type  [default: WIB]
  -m, --map_id [VDColdbox|HDColdbox|ProtoDUNESP1|PD2HD|VST]
                                  Select input channel map  [default:
                                  HDColdbox]
  -t, --threshold INTEGER         Enter threshold used in run  [default: 100]
  -w, --num-waves INTEGER         Number of 1D waveforms to plot  [default:
                                  10]
  -s, --step INTEGER              Number of TPs to skip when doing 1D plots
                                  [default: 150]
  -o, --outpath TEXT              Output path for plots  [default: .]
  -h, --help                      Show this message and exit.
```

The usual way to run it when dealing with a TR file would be:
```
dtp-feedback-plots.py <FILES_PATH> --input_type TR --tr-num <tr-num> --frame_type <frame_type> --map_id <map_id> --threshold <threshold> --num-waves <num-waves> --step <step> --output <out_path> --interactive
```

This command will print in the terminal the pandas DataFrames containing the FW TPs and TPs from the specified trigger number. A typical DataFrame containing FW TPs should look like this:
```
                     ts  offline_ch  crate_no  slot_no  fiber_no  wire_no  flags  median  accumulator  start_time  end_time  peak_time  peak_adc  hit_continue  tp_flags  sum_adc
0    104174045922344937        6679         1        4         0      207      0    8946           -1           4        19         11     10056             0         0    40547
1    104174045922345481        6356         1        3         1      204      0    8749           -1           4        19         11      9828             0         0    38152
2    104174045922345481        6357         1        3         1      205      0    8899           -1           4        19         12      9199             0         0    31710
3    104174045922345481        6358         1        3         1      206      0    8787            4           4        20         11     10157             0         0    43372
4    104174045922346985        6679         1        4         0      207      0    8945            0           4        19         11     10098             0         0    41177
..                  ...         ...       ...      ...       ...      ...    ...     ...          ...         ...       ...        ...       ...           ...       ...      ...
507  104174045922613481        5956         1        0         1      204      0    8874           -3           3        20         11     10041             0         0    43984
510  104174045922615529        5959         1        0         1      207      0    8823            4           4        20         12     10131             0         0    42930
508  104174045922615529        5957         1        0         1      205      0    8726            3           4        19         11      9447             0         0    34221
509  104174045922615529        5958         1        0         1      206      0    8889            2           4        19         11      9985             0         0    39855
511  104174045922615529        5956         1        0         1      204      0    8874            2           4        20         11      9955             0         0    41241

[1408 rows x 16 columns]
```
whereas a DataFrame with TPs:
```
              start_time           peak_time  time_over_threshold  offline_ch  sum_adc  peak_adc  flag
0     104174045922341200  104174045922341488                  512        5961    41338      9950     0
1     104174045922341200  104174045922341456                  544        5962    45728     10233     0
2     104174045922341232  104174045922341456                  480        5959    39325      9929     0
3     104174045922341232  104174045922341456                  480        5960    36040      9587     0
4     104174045922343280  104174045922343536                  480        5959    38845      9883     0
...                  ...                 ...                  ...         ...      ...       ...   ...
1403  104174045922599568  104174045922599824                  480        6361    39621      9970     0
1404  104174045922601072  104174045922601328                  480        6679    38554      9850     0
1405  104174045922601072  104174045922601296                  480        6680    34597      9463     0
1406  104174045922601072  104174045922601296                  480        6681    39015      9873     0
1407  104174045922601072  104174045922601296                  512        6682    42105     10035     0

[1408 rows x 7 columns]
```

The script also outputs four pdf files. The first two contain 2D event displays of the data from the corresponding trigger, with the upper panel showing the raw ADCs (with a fast pedestal subtraction applied) and the lower panel overlaying the position of either the FW TPs (distinguishing between hit continue 0 and 1)or the TPs. A normal TR display with FW TPs should look like this:
![TRDisplay_fwtp_ex](https://user-images.githubusercontent.com/73996651/199078780-ac260b4e-d426-4096-b2cd-b6a06236372f.png)
whereas one with TPs:
![TRDisplay_tp_ex](https://user-images.githubusercontent.com/73996651/199080057-b9940539-4ac3-405d-b254-33260a2012d0.png)

The other two pdf files contain 1D plots with raw ADCs and the associated FW TPs, including some general information about the FW TP (white box), a summary of hit quantities (red box) and some derived from the raw waveform (blue box). For the first file we show FW TPs located at the centre of the packet (i.e. not touching the edges):

![hit_center_waveform](https://user-images.githubusercontent.com/73996651/199191039-3ede06a4-c195-4961-9ebf-172591de9797.png)
whereas for the second we select FW TPs on the edges (either the start or the end of the packet):
![hit_edge_waveform](https://user-images.githubusercontent.com/73996651/199191213-5315d06b-de96-4752-bac3-495028057200.png)

Additionally, these figures include a table to the side with all the information of the FW TPs falling in the plotting window.

### Exporting Trigger Record data

When using the aforementioned script with a Trigger Record file it internally unpacks the data and formats it as different pandas DataFrames as a previous step before plotting. One also has the possibility of simply do the formatting and export the resulting DataFrames to various formats. To see the different options type:
```
dtp-tr-exporter.py --help
```
which should return:
```
Usage: dtp-tr-exporter.py [OPTIONS] FILE_PATH

Options:
  -n, --tr-num INTEGER            Enter trigger number to export  [default: 1]
  -i, --interactive               Run interactive mode  [default: False]
  -f, --frame_type [ProtoWIB|WIB]
                                  Select input frame type  [default: WIB]
  -m, --map_id [VDColdbox|HDColdbox|ProtoDUNESP1|PD2HD|VST]
                                  Select input channel map  [default:
                                  HDColdbox]
  --out_format [HDF5|CSV]         Select format of output  [default: HDF5]
  -o, --out_path TEXT             Output path for plots  [default: .]
  -h, --help                      Show this message and exit.
```

The normal use this script would be something like:
```
dtp-tr-exporter.py <FILES_PATH> --tr-num <tr-num> --frame_type <frame_type> --map_id <map_id> --out_format <out_format> --output <out_path> --interactive
```

Selecting as output format `HDF5` will create a single hdf5 file containing four pandas DataFrames:
1. `info` contains information about the Trigger Record.
2. `raw_fwtps` contains the FW TPs.
3. `raw_adcs` contains the raw ADCs.
4. `tps` contains the TPs.

With the option `CSV` one gets four different csv files, with the same naming convention as the keys in the hdf5 files, containing the same data.

Exporting the data to hdf5 (or csv) is useful for running the plotting scripts or to performing further analysis with it faster. Data in hdf5 files containing these DataFrames can also be used as input for the plotting script, just need to select the adequate option for input type:
```
dtp-feedback-plots.py <FILES_PATH> --input_type DF --tr-num <tr-num> --frame_type <frame_type> --map_id <map_id> --threshold <threshold> --num-waves <num-waves> --step <step> --output <out_path> --interactive
```

## Feedback from Binary Dumps