#!/bin/bash

input_msl='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/era5_3hourly.nc'
input_vo='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/vort_subset_te_700.nc'

DetectNodes --in_data "$input_msl;$input_vo" --timefilter "3hr" --out "cyclone_DN.txt" --searchbymin 'msl' --closedcontourcmd "msl,200.0,6.0,0" --mergedist 6.0 --thresholdcmd "vo,>,1e-05,1.0" --outputcmd "msl,min,0" --regional --latname "latitude" --lonname "longitude" 

StitchNodes --in "cyclone_DN.txt" --out "cyclone_SN.txt" --range 6.0 --min_endpoint_dist 8.0 --maxgap 4