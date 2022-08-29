#!/bin/bash

#if [ $# -eq 0 ]; then
#    echo "No arguments provided"
#    exit 1
#fi
#IO/earlyEst/2017_0720_kos-bodrum_stat.json

#initialization
event=2017_0720_kos-bodrum
mainFolder=$(pwd)
config=$mainFolder/../cfg/ptf_main.config
eventfile=$mainFolder/../IO/earlyEst/$event\_stat.json
workdir=$mainFolder/IO/$domain\_$eventID

enqueue_compss \
--pythonpath=/home/bsc44/bsc44973/Codes/ptf_core/Step1_EnsembleDef_pycompss \
--job_name=step1 \
--output_profile=output_profiles/profile.txt \
--graph=true \
--num_nodes=1 \
--qos=debug \
--exec_time=10 \
--lang=python \
./launch_step1_pycompss.py --cfg $config --event $eventfile
