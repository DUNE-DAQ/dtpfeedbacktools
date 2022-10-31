# dtpfeedbacktools - Tools for DTP prompt feedback

Simple package developed to validate the recorded data and firmware functioning. Data can be visualised and quality inspected almost instantaneously after capture. Some of the features it offers are:

1. Quick measurement of usable overlap in data captures (binary dumps).
2. Detailed comparison for raw data, FW TPs and TPs on individual channels.
3. 2D event displays to check the overall quality of the capture.
4. Possible emulation of the TPG block ([dtpemulator](https://github.com/DUNE-DAQ/dtpemulator)).

## Requirements
A set of particular Python packages are required to run these tools. Make sure to include them in your environment by running:
```
pip install -r requirements.txt
```

## Feedback from Trigger Record files

The main inputs for validation are currently Trigger Records (TRs). These contain the raw ADC data, firmware TPs and TPs. After a run is taken and the TRs are writen one would like to run `dtp-feedback-plots.py`. To see the different inputs you can provide to it type:
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
This command will print in the terminal the pandas DataFrames containing the FW TPs and TPs from the especified trigger number.

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
```

## Feedback from Binary Dumps


