import os
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import *
from PyOphidia import cube, client
import numpy as np

#This task imports the file NetCDF with the bathymetry variable into Ophidia
@task(returns=object)
def importBathymetry():
    cube.Cube.setclient(read_env=True, project="0459")
    sourcePath = "Step2_BS/"
    bath = cube.Cube.importnc2(
        src_path= sourcePath + 'bathymetry.nc',
        measure='deformed_bathy',
        imp_dim='time',
        exp_dim='grid_npoints',
        check_compliance='no',
        nhost=4,
        ncores=4,
        nfrag=1
        )
    bath_reduce=bath.reduce(operation="max")
    depth = bath_reduce.apply(query="oph_math('OPH_FLOAT','OPH_FLOAT',oph_math('OPH_FLOAT','OPH_FLOAT',oph_predicate('OPH_FLOAT','OPH_FLOAT',measure,'x-1','<0','1','x'),'OPH_MATH_SQRT'),'OPH_MATH_SQRT')")
    return depth

#This task creates an Ophidia container for each scenario
@task(returns=str)
def createContainer(group):
    cube.Cube.setclient(read_env=True, project="0459")
    container='scenarios' + str(group)
    cube.Cube.createcontainer(container=container,dim='grid_npoints|time',dim_type='double|double',hierarchy='oph_base|oph_time')
    return container

#This task imports the NetCDF file with the eta variable into Ophidia
@task(returns=object)
def OphidiaImport(scenario, depth, container):
    cube.Cube.setclient(read_env=True, project="0459")
    sourcePath = "Step2_BS/"
    filePath = sourcePath + 'BS_scenario' + scenario + '/out_ts.nc'
    # Check if a file exists
    while True:
        if os.path.exists(filePath):
            break
    ts = cube.Cube.importnc2(
        container=container,
        src_path= filePath,
        measure='eta',
        imp_dim='time',
        exp_dim='grid_npoints',
        check_compliance='no',
        nhost=4,
        ncores=4,
        nfrag=1
        )
    return ts

#This task computes the ts_max
@task(returns=str)
def OphidiaMax(ts):
    cube.Cube.setclient(read_env=True, project="0459")
    ts_max = ts.reduce(operation="max")
    return ts_max.pid

#This task computes the ts_min
@task(returns=str)
def OphidiaMin(ts):
    cube.Cube.setclient(read_env=True, project="0459")
    ts_max = ts.reduce(operation="min")
    return ts_max.pid

#This task computes the ts_off
@task(returns=object)
def OphidiaOffset(ts):
    cube.Cube.setclient(read_env=True, project="0459")
    # Computation of ts_offset
    firstRow=ts.subset(subset_dims="time",subset_filter="1",subset_type="index")
    ts0 = firstRow.apply(query="oph_extend('OPH_FLOAT','OPH_FLOAT',measure,961)")
    ts_off = ts.intercube(cube2=ts0.pid, operation='sub')
    return ts_off

#This task computes the Peak to through
@task(returns=str)
def OphidiaP2t(ts_max_pid, ts_min_pid):
    cube.Cube.setclient(read_env=True, project="0459")
    ts_max=cube.Cube(pid=ts_max_pid)
    diff = ts_max.intercube(cube2=ts_min_pid,operation="sub")
    ts_p2t = diff.apply(query="oph_mul_scalar('OPH_FLOAT','OPH_FLOAT',measure,0.5)")
    return ts_p2t.pid

#This task applies the Green Law
@task(returns=str)
def OphidiaGL(cube_pid, depth):
    cube.Cube.setclient(read_env=True, project="0459")
    ts_cube=cube.Cube(pid=cube_pid)
    gl_cube = ts_cube.intercube(cube2=depth.pid,operation="mul")
    return gl_cube.pid

#This task merges all datacubes adding the scenario dimension and saves a NetCDF file for each calculated variable renaming its name
@task(returns=object, cubes=COLLECTION_IN)
def OphidiaMerge(cubes, name, group):
    cube.Cube.setclient(read_env=True, project="0459")
    pids = '|'.join(cubes)
    ts_merge = cube.Cube.mergecubes2(cubes=pids, dim="scenarios")
    ts_merge.exportnc2(output_path="tsunami",output_name=name+str(group))
    cube.Cube.script(script='tsunami/rename_variables_group.sh',args=name+'|'+str(group)+'|tsunami/')
    return ts_merge

#This task creates a NetCDF file merging all previous files together
@task(dependencies=COLLECTION_IN)
def OphidiaCDOMerge(group, scenarios, container, dependencies):
    cube.Cube.setclient(read_env=True, project="0459")
    cube.Cube.script(script='tsunami/merge_files_group.sh', args=str(group)+'|'+str(scenarios)+'|tsunami/')
    cube.Cube.deletecontainer(container=container,force='yes')

# INITIALIZE
cube.Cube.setclient(read_env=True, project="0459")

try:
    cube.Cube.cluster(action='deploy', host_partition="test", nhost=4,exec_mode='async')
except:
    pass


# Invoking tasks
if __name__ == '__main__':
    #Import bathymetry
    depth = importBathymetry()
    group=0
    scenarios=72
    for j in range(0,864,72):
        group+=1
        container = createContainer(group)
        datacubes = [0 for i in range(scenarios)]
        ts_max_cubes = [0 for i in range(scenarios)]
        ts_min_cubes = [0 for i in range(scenarios)]
        ts_off_cubes = [0 for i in range(scenarios)]
        ts_max_off_cubes = [0 for i in range(scenarios)]
        ts_min_off_cubes = [0 for i in range(scenarios)]
        ts_p2t_cubes = [0 for i in range(scenarios)]
        ts_max_gl_cubes = [0 for i in range(scenarios)]
        ts_max_off_gl_cubes = [0 for i in range(scenarios)]
        ts_p2t_gl_cubes = [0 for i in range(scenarios)]
        for i in range(scenarios):
            #Import of all scenarios files
            datacubes[i] = OphidiaImport(str(i+j+1).zfill(3),depth, container)
            ts_max_cubes[i] = OphidiaMax(datacubes[i])
            ts_min_cubes[i] = OphidiaMin(datacubes[i])
            ts_off_cubes[i] = OphidiaOffset(datacubes[i])
            ts_max_off_cubes[i] = OphidiaMax(ts_off_cubes[i])
            ts_min_off_cubes[i] = OphidiaMin(ts_off_cubes[i])
            ts_p2t_cubes[i] = OphidiaP2t(ts_max_cubes[i], ts_min_cubes[i])
            ts_max_gl_cubes[i] = OphidiaGL(ts_max_cubes[i], depth)
            ts_max_off_gl_cubes[i] = OphidiaGL(ts_max_off_cubes[i], depth)
            ts_p2t_gl_cubes[i] = OphidiaGL(ts_p2t_cubes[i], depth)
        #Merge all scenarios
        ts_max_cube = OphidiaMerge(ts_max_cubes, "max", group)
        ts_min_cube = OphidiaMerge(ts_min_cubes, "min", group)
        ts_max_off_cube = OphidiaMerge(ts_max_off_cubes, "max_off", group)
        ts_min_off_cube = OphidiaMerge(ts_min_off_cubes, "min_off", group)
        ts_p2t_cube = OphidiaMerge(ts_p2t_cubes, "p2t", group)
        ts_max_gl_cube = OphidiaMerge(ts_max_gl_cubes, "max_gl", group)
        ts_max_off_gl_cube = OphidiaMerge(ts_max_off_gl_cubes, "max_off_gl", group)
        ts_p2t_gl_cube = OphidiaMerge(ts_p2t_gl_cubes, "p2t_gl", group)
        #Merge all files in a single output file
        OphidiaCDOMerge(group, scenarios, container, [ts_max_cube, ts_min_cube, ts_max_off_cube, ts_min_off_cube, ts_p2t_cube, ts_max_gl_cube, ts_max_off_gl_cube, ts_p2t_gl_cube])

    
