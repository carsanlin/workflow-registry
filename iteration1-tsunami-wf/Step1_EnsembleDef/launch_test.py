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
from module_test import main

#@task(event=FILE_IN)
@task()
def step1_func():
    print(main())
    pass

if __name__ == '__main__':
     print(step1_func())

