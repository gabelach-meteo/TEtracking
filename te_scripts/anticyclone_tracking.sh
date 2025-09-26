#!/bin/bash

input_msl='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/era5_3hourly.nc'
input_vo='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/vort_subset_te_700.nc'
input_z_500='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/geopotential_subset_te_500.nc'

DetectNodes --in_data "$input_msl;$input_z_500" --timefilter "3hr" --out "anticyclone_DN.txt" --searchbymax 'msl' --closedcontourcmd "msl,-75.0,12.0,0;z,-20.0,12.0,0" --mergedist 6.0 --outputcmd "msl,min,0" --regional --latname "latitude" --lonname "longitude" 

StitchNodes --in "anticyclone_DN.txt" --out "anticyclone_SN.txt" --range 6.0 --min_endpoint_dist 2.0 --maxgap 4