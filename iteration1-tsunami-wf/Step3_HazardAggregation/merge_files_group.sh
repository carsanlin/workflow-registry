#!/bin/bash
group="$1"
scenarios="$2"
mul=$(((group-1)*scenarios))
path="$3"
file="Step2_BS_hmax${group}.nc"
filescenarios="Step2_BS_hmax_scenarios${group}.nc"
if [ -f "$path$file" ]; then
    rm $path$file
fi
if [ -f "$path$filescenarios" ]; then
    rm $path$filescenarios
fi
input_files="${path}ts_max${group}.nc ${path}ts_min${group}.nc ${path}ts_max_off${group}.nc ${path}ts_min_off${group}.nc ${path}ts_p2t${group}.nc ${path}ts_max_gl${group}.nc ${path}ts_max_off_gl${group}.nc ${path}ts_p2t_gl${group}.nc"
cdo merge $input_files ${path}${file}; 
rm $input_files
ncap2 -s 'scenarios=scenarios+'$mul ${path}${file} ${path}${filescenarios}
