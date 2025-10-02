#!/bin/bash

input_msl='/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/era5-modified/NH/msl_fixed.nc'

DetectNodes --in_data "$input_msl" --timefilter "3hr" --out "/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/TEtextfiles/DetectNodes/NH/c_DN_f1.txt" --searchbymin 'msl' --closedcontourcmd "msl,200.0,8.0,0" --mergedist 8.0 --outputcmd "msl,min,0" --regional --latname "latitude" --lonname "longitude" 

StitchNodes --in "/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/TEtextfiles/DetectNodes/NH/c_DN_f1.txt" --out "/home/glach/projects/def-rfajber/shared/gabe-fall-2025/TEtracking/data/TEtextfiles/StitchNodes/NH/c_SN_f1.txt" --range 8.0 --min_endpoint_dist 8.0 --maxgap 4 --mintime "24h"