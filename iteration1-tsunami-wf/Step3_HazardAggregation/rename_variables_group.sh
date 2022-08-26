#!/bin/bash
variable="$1"
group="$2"
path="$3"
input_file="${1}${2}.nc"
output_file="ts_${1}${2}.nc"
cdo chname,eta,ts_$variable $path$input_file $path$output_file;
rm $path$input_file
