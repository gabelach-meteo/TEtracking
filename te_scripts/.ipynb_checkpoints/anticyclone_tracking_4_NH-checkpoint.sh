#!/bin/bash

input_msl='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/NH/msl_fixed.nc'


DetectNodes --in_data "$input_msl" --timefilter "3hr" --out "/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/TEtextfiles/DetectNodes/NH/anticyclone_DN_msl_200.txt" --searchbymax 'msl' --closedcontourcmd "msl,-150.0,10.0,0" --mergedist 8.0 --thresholdcmd 'msl,>,100000,0.5' --outputcmd "msl,min,0" --regional --latname "latitude" --lonname "longitude" 

StitchNodes --in "/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/TEtextfiles/DetectNodes/NH/anticyclone_DN_msl_200.txt" --out "anticyclone_SN_msl_200.txt" --range 8.0 --min_endpoint_dist 8.0 --maxgap 4