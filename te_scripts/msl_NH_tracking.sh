#!/bin/bash

input_msl='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/NH/msl_fixed.nc'

DetectNodes --in_data "$input_msl" --timefilter "3hr" --out "data/TEtextfiles/DetectNodes/NH/cyclone_DN_msl.txt" --searchbymin 'msl' --closedcontourcmd "msl,750.0,9.0,0" --mergedist 9.0 --outputcmd "msl,min,0" --regional --latname "latitude" --lonname "longitude" 

StitchNodes --in "data/TEtextfiles/DetectNodes/NH/cyclone_DN_msl.txt" --out "data/TEtextfiles/StitchNodes/NH/cyclone_SN_msl.txt" --range 8.0 --min_endpoint_dist 8.0 --maxgap 4