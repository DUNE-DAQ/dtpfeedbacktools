# dtpfeedbacktools - Tools for DTP prompt feedback

## Requirements

## Feedback scripts

**dtp-tstamp-check**

```
dtp-tstamp-check.py <FILES_PATH> -f <frame_type> -m <map_id> -o <out_path> --plots --interactive
```

**dtpft-validation-plots**


```
dtpft-validation-plots.py <FILES_PATH> -m <map_id> -n <n_lines> -o <out_path> --interactive
```

## ADC plotting tools

**dtp-fwtp-plots**


```
dtp-fwtp-plots.py <FILES_PATH> -f <frame_type> -m <map_id> -t <threshold> -n <n_plots> -o <out_path> --interactive
```

**dtp-tr-plots**


```
dtp-tr-plots.py <FILES_PATH> -f <frame_type> -m <map_id> -t <threshold> -n <n_plots> -o <out_path> --interactive
```


**dtp-evt-disp-plot**
*2D raw data event display with overlaid, color-coded, firmware hits* 

```
python dtp-evtdisp-plot.py <HDf5_FILE_PATH> -r <run_number> -n <n_timesamples_to_plot>
```
