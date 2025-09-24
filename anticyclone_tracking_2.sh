#!/bin/bash

input_msl='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/era5_3hourly.nc'
input_vo='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/vort_subset_te_700.nc'
input_z_500='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/geopotential_subset_te_500.nc'

DetectNodes --in_data "$input_msl" --timefilter "3hr" --out "anticyclone_DN_3.txt" --searchbymax 'msl' --closedcontourcmd "msl,-150.0,8.0,0" --mergedist 6.0 --thresholdcmd 'msl,>,101500,0.5' --outputcmd "msl,min,0" --regional --latname "latitude" --lonname "longitude" 

StitchNodes --in "anticyclone_DN_3.txt" --out "anticyclone_SN_3.txt" --range 6.0 --min_endpoint_dist 8.0 --maxgap 4