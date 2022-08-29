#!/usr/bin/env python
#!/home/louise/miniconda3/bin/python3.8

# Import system modules
import os
import ast
import sys
import utm
import threading
import configparser
import ray
import h5py

import numpy       as np

from time     import gmtime
from time     import strftime
from datetime import datetime

# adding the path to find some modules
#sys.path.append('Step1_EnsembleDef_python/py')
sys.path.append('../Commons/py')
# Import functions from pyPTF modules
from ptf_preload             import load_PSBarInfo
from ptf_preload             import ptf_preload
from ptf_preload             import load_Scenarios_Reg
from ptf_preload_ND          import ptf_preload_ND
from ptf_preload_ND          import load_PSBarInfo_ND
#from ptf_preload_curves      import load_hazard_values
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser              import parse_ptf_stdin
from ptf_parser              import update_cfg
#from ptf_workflow            import workflow
#from ptf_rabbit              import get_rabbit_parameters
#from ptf_rabbit              import CNTClient
#from ptf_rabbit              import send_messages
#from ptf_rabbit              import CS_to_json
#from ptf_matrix              import chck_if_point_is_land
#from ptf_matrix              import get_distance_point_to_Ring
from ptf_figures             import make_ptf_figures
from ptf_save                import save_ptf_dictionaries
from ptf_save                import save_ptf_dictionaries
from ptf_save                import save_ptf_as_txt
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import save_ptf_out
#from ptf_messages            import create_message
from ptf_ellipsoids          import build_location_ellipsoid_objects
from ptf_ellipsoids          import build_ellipsoid_objects
#from ptf_mix_utilities       import check_if_neam_event
#from ptf_mix_utilities       import merge_event_dictionaries
from ptf_mix_utilities        import conversion_to_utm
from ptf_lambda_bsps_load      import load_lambda_BSPS
from ptf_lambda_bsps_sep       import separation_lambda_BSPS
from ptf_pre_selection         import pre_selection_of_scenarios
from ptf_short_term                 import short_term_probability_distribution
from ptf_probability_scenarios      import compute_probability_scenarios
from ptf_ensemble_sampling_MC       import compute_ensemble_sampling_MC
from ptf_ensemble_sampling_IS       import compute_ensemble_sampling_IS
from ptf_ensemble_sampling_RS       import compute_ensemble_sampling_RS
#from ptf_hazard_curves         import compute_hazard_curves
#from ptf_alert_levels          import set_alert_levels

#from ptf_messages            import create_message_matrix
# from ttt                     import run_ttt
# from ttt_map_utilities       import extract_contour_times
# from ttt_map_utilities       import build_tsunami_travel_time_map

def step1_ensembleEval(**kwargs):

    Scenarios_PS     = kwargs.get('Scenarios_PS', None)
    Scenarios_BS     = kwargs.get('Scenarios_BS', None)
    LongTermInfo     = kwargs.get('LongTermInfo', None)
    ND_LongTermInfo  = kwargs.get('ND_LongTermInfo', None)
    POIs             = kwargs.get('POIs', None)
    PSBarInfo        = kwargs.get('PSBarInfo', None)
    ND_PSBarInfo     = kwargs.get('ND_PSBarInfo', None)
    Mesh             = kwargs.get('Mesh', None)
    Region_files     = kwargs.get('Region_files', None)
    ND_Region_files  = kwargs.get('ND_Region_files', None)
    args             = kwargs.get('args', None)
    Config           = kwargs.get('cfg', None)
    event_parameters = kwargs.get('event_data', None)
    sim_files = kwargs.get('sim_files', None)

    ND_POIs_init = POIs
    ND_samp=int(Config.get('Sampling','ND_samp'))  
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    IS_samp_scen=int(Config.get('Sampling','IS_samp_scen'))
    IS_samp_run=int(Config.get('Sampling','IS_samp_run'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    RS_output_scen=int(Config.get('Sampling','RS_output_scen'))

    ptf_out = dict()

    print('############## Initial ensemble #################')

    print('Build ellipsoids objects')
    ellipses = build_ellipsoid_objects(event = event_parameters,
                                       cfg   = Config,
                                       args  = args)


    print('Conversion to utm')
    LongTermInfo, POIs, PSBarInfo = conversion_to_utm(longTerm  = LongTermInfo,
                                                      Poi       = POIs,
                                                      event     = event_parameters,
                                                      PSBarInfo = PSBarInfo)

    ##########################################################
    # Set separation of lambda BS-PS
    print('Separation of lambda BS-PS')
    lambda_bsps = load_lambda_BSPS(cfg                   = Config,
                                   args                  = args,
                                   event_parameters      = event_parameters,
                                   LongTermInfo          = LongTermInfo)


    lambda_bsps = separation_lambda_BSPS(cfg              = Config,
                                         args             = args,
                                         event_parameters = event_parameters,
                                         lambda_bsps      = lambda_bsps,
                                         LongTermInfo     = LongTermInfo,
                                         mesh             = Mesh)

    #print(lambda_bsps['regionsPerPS'])
    #sys.exit()
    ##########################################################
    # Pre-selection of the scenarios
    #
    # Magnitude: First PS then BS
    # At this moment the best solution is to insert everything into a dictionary (in matlab is the PreSelection structure)
    print('Pre-selection of the Scenarios')
    pre_selection = pre_selection_of_scenarios(cfg                = Config,
                                               args               = args,
                                               event_parameters   = event_parameters,
                                               LongTermInfo       = LongTermInfo,
                                               PSBarInfo          = PSBarInfo,
                                               ellipses           = ellipses)

    if(pre_selection == False):
        ptf_out = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        status             = 'end')
        return False



    ##########################################################
    # COMPUTE PROB DISTR
    #
    #    Equivalent of shortterm.py with output: node_st_probabilities
    #    Output: EarlyEst.MagProb, EarlyEst.PosProb, EarlyEst.DepProb, EarlyEst.DepProb, EarlyEst.BarProb, EarlyEst.RatioBSonTot
    print('Compute short term probability distribution')

    short_term_probability  = short_term_probability_distribution(cfg                = Config,
                                                                  args               = args,
                                                                  event_parameters   = event_parameters,
                                                                  LongTermInfo       = LongTermInfo,
                                                                  PSBarInfo          = PSBarInfo,
                                                                  lambda_bsps        = lambda_bsps,
                                                                  pre_selection      = pre_selection)

    if(short_term_probability == True):
        ptf_out = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        status             = 'end')
        return False

    ##COMPUTE PROBABILITIES SCENARIOS: line 840
    print('Compute Probabilities scenarios')
    probability_scenarios = compute_probability_scenarios(cfg                = Config,
                                                          args               = args,
                                                          event_parameters   = event_parameters,
                                                          LongTermInfo       = LongTermInfo,
                                                          PSBarInfo          = PSBarInfo,
                                                          lambda_bsps        = lambda_bsps,
                                                          pre_selection      = pre_selection,
                                                          regions            = Region_files,
                                                          short_term         = short_term_probability,
                                                          Scenarios_PS       = Scenarios_PS)


################### Monte Carlo sampling ########################
    
    if MC_samp_scen>0: 
       print('############## Monte Carlo sampling #################')
       sampled_ensemble_MC = compute_ensemble_sampling_MC(cfg                = Config,
                                                 args               = args,
                                                 event_parameters   = event_parameters,
                                                 LongTermInfo       = LongTermInfo,
                                                 PSBarInfo          = PSBarInfo,
                                                 lambda_bsps        = lambda_bsps,
                                                 pre_selection      = pre_selection,
                                                 regions            = Region_files,
                                                 short_term         = short_term_probability,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 proba_scenarios    = probability_scenarios)
       ptf_out['new_ensemble_MC']           = sampled_ensemble_MC

################### Importance sampling ########################

    if IS_samp_scen>0:
       print('############## Importance sampling #################')
       sampled_ensemble_IS = compute_ensemble_sampling_IS(cfg                = Config,
                                                 args               = args,
                                                 event_parameters   = event_parameters,
                                                 LongTermInfo       = LongTermInfo,
                                                 PSBarInfo          = PSBarInfo,
                                                 lambda_bsps        = lambda_bsps,
                                                 pre_selection      = pre_selection,
                                                 regions            = Region_files,
                                                 short_term         = short_term_probability,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 proba_scenarios    = probability_scenarios)

       ptf_out['new_ensemble_IS']           = sampled_ensemble_IS

################### Real sampling ########################

    if RS_samp_scen>0:
       print('############## Importance sampling #################')
       sampled_ensemble_RS = compute_ensemble_sampling_RS(cfg                = Config,
                                                 args               = args,
                                                 event_parameters   = event_parameters,
                                                 LongTermInfo       = LongTermInfo,
                                                 PSBarInfo          = PSBarInfo,
                                                 lambda_bsps        = lambda_bsps,
                                                 pre_selection      = pre_selection,
                                                 regions            = Region_files,
                                                 short_term         = short_term_probability,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 proba_scenarios    = probability_scenarios)

       ptf_out['new_ensemble_RS']           = sampled_ensemble_RS
       
       if RS_output_scen>0:
          for Nid in range(RS_samp_run):
              RS_samp_scen=len(ptf_out['new_ensemble_RS'][Nid]['real_par_scenarios_bs'][:,0])
              par=np.zeros((11))
              #myfile = open('./ptf_localOutput/list_scenbs.txt', 'w')
              myfile = open(sim_files,'w')
              #myfile = open('./ptf_localOutput/list_nb%d_of_%d_scenbs.txt'%(Nid,RS_samp_scen), 'w')
              for Nscen in range(RS_samp_scen):
                  #for ipar in range(11):
                  par[:]=ptf_out['new_ensemble_RS'][Nid]['real_par_scenarios_bs'][Nscen,:]
                  myfile.write("%f %f %f %f %f %f %f %f %f %f\n"%(par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8],par[9],par[10]))
              myfile.close()

################### New discretization ########################

    if ND_samp>0:
       print('############## New discretization #################')
       
       ND_short_term_probability = {}
       ND_probability_scenarios = {}
       print(ND_samp)
   
       for i in range(ND_samp):
           
           ND_PSBarInfo_init = ND_PSBarInfo
           ND_POIs      = ND_POIs_init 
   
           print('Conversion to utm')
           ND_LongTermInfo[i], ND_POIs, ND_PSBarInfo_init = conversion_to_utm(longTerm  = ND_LongTermInfo[i],
                                                             Poi       = ND_POIs,
                                                             event     = event_parameters,
                                                             PSBarInfo = ND_PSBarInfo_init)
   
           ##########################################################
           # Set separation of lambda BS-PS
           print('Separation of lambda BS-PS')
           ND_lambda_bsps = load_lambda_BSPS(cfg                   = Config,
                                          args                  = args,
                                          event_parameters      = event_parameters,
                                          LongTermInfo          = ND_LongTermInfo[i])
   
   
           ND_lambda_bsps = separation_lambda_BSPS(cfg              = Config,
                                                   args             = args,
                                                   event_parameters = event_parameters,
                                                   lambda_bsps      = ND_lambda_bsps,
                                                   LongTermInfo     = ND_LongTermInfo[i],
                                                   mesh             = Mesh)
   
           #print(lambda_bsps['regionsPerPS'])
           #sys.exit()
           ##########################################################
           # Pre-selection of the scenarios
           #
           # Magnitude: First PS then BS
           # At this moment the best solution is to insert everything into a dictionary (in matlab is the PreSelection structure)
           print('Pre-selection of the Scenarios')
           ND_pre_selection = pre_selection_of_scenarios_ND(cfg                = Config,
                                                      args               = args,
                                                      event_parameters   = event_parameters,
                                                      LongTermInfo       = ND_LongTermInfo[i],
                                                      Ori_LongTermInfo   = LongTermInfo,
                                                      PSBarInfo          = ND_PSBarInfo_init,
                                                      ellipses           = ellipses)
   
           if(ND_pre_selection == False):
           #    ptf_out = save_ptf_dictionaries(cfg                = Config,
           #                                    args               = args,
           #                                    event_parameters   = event_parameters,
           #                                    status             = 'end')
               return False
   
           # COMPUTE PROB DNDTR
           #
           #    Equivalent of shortterm.py with output: node_st_probabilities
           #    Output: EarlyEst.MagProb, EarlyEst.PosProb, EarlyEst.DepProb, EarlyEst.DepProb, EarlyEst.BarProb, EarlyEst.RatioBSonTot
           print('Compute short term probability distribution')
           ND_short_term_probability[i] = dict()
           ND_short_term_probability[i] = short_term_probability_distribution_ND(cfg                = Config,
                                                                         args               = args,
                                                                         event_parameters   = event_parameters,
                                                                         LongTermInfo       = ND_LongTermInfo[i],
                                                                         PSBarInfo          = ND_PSBarInfo_init,
                                                                         lambda_bsps        = ND_lambda_bsps,
                                                                         pre_selection      = ND_pre_selection)
   
           if(short_term_probability == True):
           #    ptf_out = save_ptf_dictionaries(cfg                = Config,
           #                                    args               = args,
           #                                    event_parameters   = event_parameters,
           #                                    status             = 'end')
               return False
   
           ##COMPUTE PROBABILITIES SCENARIOS: line 840
           print('Compute Probabilities scenarios')
           ND_probability_scenarios[i] = dict()
           ND_probability_scenarios[i] = compute_probability_scenarios_ND(cfg                = Config,
                                                                 args               = args,
                                                                 event_parameters   = event_parameters,
                                                                 LongTermInfo       = ND_LongTermInfo[i],
                                                                 Ori_LongTermInfo       = LongTermInfo,
                                                                 PSBarInfo          = ND_PSBarInfo_init,
                                                                 lambda_bsps        = ND_lambda_bsps,
                                                                 pre_selection      = ND_pre_selection,
                                                                 Ori_pre_selection      = pre_selection,
                                                                 regions            = ND_Region_files[i],
                                                                 short_term         = ND_short_term_probability[i],
                                                                 ori_prob_scen      = probability_scenarios,
                                                                 Scenarios_PS       = Scenarios_PS)
           
           ptf_out['ND_short_term_probability'] = ND_short_term_probability
           ptf_out['ND_probability_scenarios']  = ND_probability_scenarios     



    if(probability_scenarios == False):
        print( "--> No Probability scenarios found. Save and Exit")
        ptf_out = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        status             = 'end')
        return False


    # in order to plot here add nested dict to ptf_out
    ptf_out['short_term_probability'] = short_term_probability
    ptf_out['event_parameters']       = event_parameters
    ptf_out['probability_scenarios']  = probability_scenarios
    ptf_out['POIs']                   = POIs

    print('End pyPTF')

    return ptf_out



################################################################################################
#                                                                                              #
#                                  BEGIN                                                       #
################################################################################################

#def run_step1_init(config,event):
def run_step1_init(args,sim_files):

    ############################################################
    # Read Stdin
    #print('\n')
    #args=parse_ptf_stdin()
    #args           = kwargs.get('args', None)    

    ############################################################
    # Initialize and load configuaration file
    #cfg_file        = args.cfg
    #cfg_file        = kwargs.get('config', None)
    #event           = kwargs.get('event', None)
    cfg_file        = args.cfg

    Config          = configparser.RawConfigParser()
    Config.read(cfg_file)
    Config          = update_cfg(cfg=Config, args=args)
    min_mag_message = float(Config.get('matrix','min_mag_for_message'))
    ND_samp=int(Config.get('Sampling','ND_samp'))    

    pwd = os.getcwd()
    print('pwd',pwd) 
    ############################################################
    #LOAD INFO FROM SPTHA
    PSBarInfo                                         = load_PSBarInfo(cfg=Config, args=args)
    # hazard_curves_files                               = load_hazard_values(cfg=Config, args=args, in_memory=True)
    Scenarios_PS, Scenarios_BS                        = load_Scenarios_Reg(cfg=Config, args=args, in_memory=True)
    LongTermInfo, POIs, Mesh, Region_files            = ptf_preload(cfg=Config, args=args)
    
    begin_of_time = datetime.utcnow()
    
    # gaga='/data/pyPTF/hazard_curves/glVal_BS_Reg032-E02352N3953E02776N3680.hdf5'
    
    # with h5py.File(gaga, "r") as f:
    #     a_group_key = list(f.keys())[0]
    #     datagaga = np.array(f.get(a_group_key))
    # print(np.shape(datagaga), datagaga.nbytes)
    end_of_time = datetime.utcnow()
    diff_time        = end_of_time - begin_of_time
    print("--> Execution Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds))
    #sys.exit()
    
    
    begin_of_time = datetime.utcnow()
    
    #### Load event parameters then workflow and ttt are parallel
    #print('############################')
    print('Load event parameters')
    # Load the event parameters from json file consumed from rabbit
    #event        = kwargs.get('event', None)
    event_parameters = load_event_parameters(event       = args.event,
                                             format      = args.event_format,
                                             routing_key = 'INT.QUAKE.CAT',
                                             args        = args,
                                             json_rabbit = None,
                                             cfg         = Config)
    print_event_parameters(dict=event_parameters, args = args)
    
    #ee_d=event_parameters
    #PosCovMat_2d = np.array([[ee_d['cov_matrix']['XX'], ee_d['cov_matrix']['XY']], \
    #                                 [ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['YY']]])
    #PosMean_2d      = np.array([ee_d['ee_utm'][1], \
    #                                 ee_d['ee_utm'][0]])
    #PosCovMat_3d    = np.array([[ee_d['cov_matrix']['XX'], ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['XZ']], \
    #                                 [ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['YY'], ee_d['cov_matrix']['YZ']], \
    #                                 [ee_d['cov_matrix']['XZ'], ee_d['cov_matrix']['YZ'], ee_d['cov_matrix']['ZZ']]])
    #
    #PosCovMat_3dm    = PosCovMat_3d*100000
    #
    #s = open(args.event, 'r').read()
    #jsn_object = eval(s)
    #json_string = eval(s)
    #
    ## Epicenter informations
    #lon           =  float(json_string['features'][0]['geometry']['coordinates'][0])
    #lat           =  float(json_string['features'][0]['geometry']['coordinates'][1])
    #depth         =  float(json_string['features'][0]['geometry']['coordinates'][2])
    #area          =  str(json_string['features'][0]['properties']['place'])
    #OT            =  str(json_string['features'][0]['properties']['time'])
    #mag           =  float(json_string['features'][0]['properties']['mag'])
    #ev_type       =  str(json_string['features'][0]['properties']['type'])
    #mag_type      =  str(json_string['features'][0]['properties']['magType'])
    #
    ##utm conversion
    #ee_utm = utm.from_latlon(lat, lon)
    #
    #PosMean_3d      = np.array([lat, \
    #                            lon, \
    #                            depth * 1000.0])
    #
    #PosMean_3dm      = np.array([ee_utm[0], \
    #                            ee_utm[1], \
    #                            depth * 1000.0])
    #
    #print(PosCovMat_3d)
    #print("    ")
    #print(PosCovMat_3dm)
    #print("    ")
    #print(PosMean_3d)
    #print("    ")
    #print(PosMean_3dm)
    #print("    ")
    
    
    #### Load parameters and data for the new discretization (ND) ####
    ND_samp=int(Config.get('Sampling','ND_samp'))
    ND_LongTermInfo={}
    ND_Region_files={}
    ND_PSBarInfo = load_PSBarInfo_ND(cfg=Config, args=args)
    for i in range(ND_samp):
       ND_LongTermInfo[i] = dict()
       ND_Region_files[i] = dict()
       ND_LongTermInfo[i],ND_Region_files[i] = ptf_preload_ND(cfg=Config, 
                                                              args=args,
                                                              event_parameters   = event_parameters)
    
    # --------------------------------------------------- #
    # check  inneam, mag_action are true or false
    # Qui controlla se la magnitudo minima indicata dalla matrice decisionale e' raggiunta
    # e se l'evento si trova nell'area neam. nel primo caso, se la magnitudo e' sotto soglia,
    # la procedura invia un messaggio al rabbit contentente l'informazione 'magnitude < 5.5, nothing to do'
    # che verra mostrata su jet.
    # La differenza tra area neam o fuori, per ora e' solo legata alla differenza dei forecast point da caricare
    # Le differenze di invio sono gestite da catcom@tigerX
    # --------------------------------------------------- #
    #event_parameters = check_if_neam_event(dictionary=event_parameters, cfg=Config)
    #print(" --> Event INNEAM:            ", event_parameters['inneam'])
    
    
    # --------------------------------------------------- #
    # check if event inland with respect neam guidelines
    # event_parameters, geometry_land_with_point         = chck_if_point_is_land(event_parameters=event_parameters, cfg=Config)
    # event_parameters['epicentral_distance_from_coast'] = get_distance_point_to_Ring(land_geometry=geometry_land_with_point, event_parameters=event_parameters)
    ptf_out = save_ptf_dictionaries(cfg                = Config,
                                    args               = args,
                                    event_parameters   = event_parameters,
                                    status             = 'new')
    
    
    
    #print(" --> Epicenter INLAND:           %r  (%.3f [km]) " % (event_parameters['epicenter_is_in_land'],  event_parameters['epicentral_distance_from_coast'] ))
    
    
    
    #nr_cpu_allowed        = float(Config.get('Settings','nr_cpu_max'))
    #print(" --> Initialize ray with %d cpu" % (int(nr_cpu_allowed)))
    #ray.init(num_cpus=int(nr_cpu_allowed), include_dashboard=False, ignore_reinit_error=True)
    #print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        
    ######################################################
    # Ensemble evaluation
    
    
    ptf_out = step1_ensembleEval(Scenarios_PS = Scenarios_PS,
                                            Scenarios_BS = Scenarios_BS,
                                            LongTermInfo = LongTermInfo,
                                            ND_LongTermInfo = ND_LongTermInfo,
                                            POIs         = POIs,
                                            PSBarInfo    = PSBarInfo,
                                            ND_PSBarInfo    = ND_PSBarInfo,
                                            Mesh         = Mesh,
                                            Region_files = Region_files,
                                            ND_Region_files = ND_Region_files,
                                            #h_curve_files= hazard_curves_files,
                                            args         = args,
                                            cfg          = Config,
                                            event_data   = event_parameters,
                                            sim_files    = sim_files)
    
    
    ######################################################
    # Save outputs
    print("Save pyPTF output")
    saved_files = save_ptf_out(cfg                = Config,
                               args               = args,
                               event_parameters   = event_parameters,
                               ptf                = ptf_out,
                               #status             = status,
                               )
    
    #saved_files = save_ptf_dictionaries(cfg                = Config,
    #                                        args               = args,
    #                                        event_parameters   = event_parameters,
    #                                        ptf                = ptf_out,
    #                                        #status             = status,
    #                                        )
    #
    #
    #######################################################
    ## Make figures from dictionaries
    #print("Make pyPTF figures")
    #saved_files = make_ptf_figures(cfg                = Config,
    #                                   args               = args,
    #                                   event_parameters   = event_parameters,
    #                                   ptf                = ptf_out,
    #                                   saved_files        = saved_files)
    #
    #print("Save some extra usefull txt values")
    #saved_files = save_ptf_as_txt(cfg                = Config,
    #                                  args               = args,
    #                                  event_parameters   = event_parameters,
    #                                  ptf                = ptf_out,
    #                                  #status             = status,
    #                                  pois               = ptf_out['POIs'],
    #                                  #alert_levels       = ptf_out['alert_levels'],
    #                                  saved_files        = saved_files,
    #                                  #fcp                = fcp_merged,
    #                                  ensembleYN         = True
    #                                  )
    #
    #
    #
