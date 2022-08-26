import sys
import os

from pycompss.api.task import task
from pycompss.api.mpi import mpi
from pycompss.api.binary import binary
from pycompss.api.constraint import constraint
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on_file

tsunamiHySEA_bin="/home/bsc44/bsc44660/PROJECTS/ITERATION1-TSUNAMI-WF/Step2_RunSimulations/IO/KosBodrum_test/TsunamiHySEA"
#tsunamiHySEA_bin="echo"
simulBS_bin="/home/bsc44/bsc44660/PROJECTS/ITERATION1-TSUNAMI-WF/Step2_RunSimulations/IO/KosBodrum_test/Step2_config_simul.sh"

gpus_per_node=4

## execution of simulator thysea using pycompss. (test)
@constraint(processors=[{'processorType':'CPU', 'computingUnits':'1'},
                        {'processorType':'GPU', 'computingUnits':'1'}])
@mpi(binary=tsunamiHySEA_bin, params="{{file_in}}", runner="mpirun", processes=gpus_per_node, processes_per_node=gpus_per_node,working_dir="/home/bsc44/bsc44660/PROJECTS/ITERATION1-TSUNAMI-WF/Step2_RunSimulations/IO/KosBodrum_test/")
@task(file_in=FILE_IN)
def mpi_func(file_in, wdir):
     pass

@binary(binary=simulBS_bin, working_dir="/home/bsc44/bsc44660/PROJECTS/ITERATION1-TSUNAMI-WF/Step2_RunSimulations/IO/KosBodrum_test/")
@task(sim_files=FILE_OUT)
def build_structure(seistype, grid, hours, group, sim_files, top_dir):
     pass

if __name__ == '__main__':
 seistype=sys.argv[1]
 grid=sys.argv[2]
 hours=sys.argv[3]
 sims_file=sys.argv[4]
 top_dir = os.getcwd()
 print(top_dir)
 ### STEP 1 ###
 build_structure(seistype, grid, hours, gpus_per_node, sims_file, top_dir)
 compss_wait_on_file(sims_file)
 with open(sims_file) as f:
     for line in f:
        # load balancing
        line=line.strip()
        print("Submitting execution for " +  line + ".")
        wdir = os.path.dirname(line)
        fout = os.path.join(wdir,"out_ts.nc")
        mpi_func(line,wdir)
###STEP3###
###STEP4###
