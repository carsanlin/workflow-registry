import sys
import os
import ast
import utm
import threading
import configparser

from pycompss.api.task import task
from pycompss.api.mpi import mpi
from pycompss.api.binary import binary
from pycompss.api.constraint import constraint
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on_file

sys.path.append('./')
from run_step1 import run_step1_init 
#from run_test import main

#@task(event=FILE_IN)
@task(sim_files=FILE_OUT)
def step1_func(config,event,sim_files):
    run_step1_init(config=config,event=event) 
    pass

if __name__ == '__main__':
     config=sys.argv[1]
     event=sys.argv[2]
     print(config)
     print(event)
     #sim_files="/logfiles/scenarios_list.txt"
     sim_files="./ptf_localOutput/list_scenbs.txt"
     step1_func(config,event,sim_files)

