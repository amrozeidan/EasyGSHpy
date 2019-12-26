#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:59:03 2019

@author: amrozeidan
"""
import sys
from os import path
import os
from scipy.spatial.distance import cdist
import numpy as np
import datetime
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib.offsetbox import AnchoredText


lib_func_fol = '/Users/amrozeidan/Documents/hiwi/easygshpy/lib_func'
commonfol = '/Users/amrozeidan/Documents/hiwi/easygshpy/com'
basefol = '/Users/amrozeidan/Documents/hiwi/easygshpy/base'
stationsDB = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/info_all_stations.dat'
slffile = '/Users/amrozeidan/Downloads/EAZYgsh tools/t2d___res2519_NSea_rev04u_y2015surge_conf10.slf'
#variables for 2d modules
reqvar = ['U' , 'V' , 'S' , 'SLNT' , 'W' ,  'A' , 'G' , 'H']
telmod = '2D'
#slffile = '/Users/amrozeidan/Downloads/EAZYgsh tools/twc___res2519_NSea_rev04u_y2015surge_conf10.slf'
#variables for tomawac module
#reqvar = ['DMOY' , 'HM0' , 'TPD' , 'TM02']
#telmod = 'TOMAWAC'


def a_extelmac(lib_func_fol , commonfol , basefol , stationsDB , slffile , reqvar , telmod):
    
    startt = time.time()
 
    sys.path.append(path.join(lib_func_fol , 'pputils' , 'ppmodules'))
    sys.path.append(path.join(lib_func_fol , 'gshtools')) 
    from selafin_io_pp import ppSELAFIN
    from TelDict import teldict
    
    
    #extracting data from all stations database
    stations_data = np.loadtxt(stationsDB , delimiter=',',skiprows=1 , dtype = str)
    
    station_names = stations_data[:,0]
    stations = station_names.tolist()
    easting = stations_data[:,1].astype(float)
    northing = stations_data[:,2].astype(float)
    
    RW_HW = np.transpose(np.vstack((easting , northing)))
    
    
    # read *.slf file
    slf = ppSELAFIN(slffile)
    slf.readHeader()
    slf.readTimes()
    
    
    #slf variables
    vnames = slf.getVarNames()
    variables = [ s.strip() for s in vnames] #remove whitespaces
    #slf units of variables
    vunits = slf.getVarUnits()
    units = [ s.strip() for s in vunits]
    #precision 
    float_type,float_size = slf.getPrecision()
    
    NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = slf.getMesh()
    
    # store slf info into arrays
    times = slf.getTimes()
    init_date = datetime.datetime(slf.DATE[0],slf.DATE[1],slf.DATE[2],slf.DATE[3],slf.DATE[4],slf.DATE[5])
    #creating date/time array
    date_array = np.array([init_date + datetime.timedelta(seconds = i ) for i in times])   
    
    #in case initial date is not defined in slf file, for the last slf file tested, something was
    #wrong with times so date_array is calculated based on DT 
#    times = slf.getTimes()
#    DT = (times[2] - times[1])/ 3600 #timestep in hours
#    init_date = datetime.datetime(2015 , 1, 6 , 0 , 0)
#    nsteps = len(slf.getTimes())
#    date_array = []
#    for i in range(nsteps):
#        date_array_i = init_date + i*datetime.timedelta(hours = DT )
#        date_array.append(date_array_i) 
    
    #defining indices of stations based on coordinates 
    XYZ = np.transpose(np.vstack((x , y)))
    D = cdist(XYZ , RW_HW)
    
    #min in each array column
    (indices , indx) = np.where(D == np.min(D , axis=0) )
    sorted_indices = [indices for _, indices in sorted(zip(indx , indices))]
    
    #preparation of the result 3d array:
    #locations * time steps * variables
    result = np.empty((len(RW_HW), len(times), len(vnames)))
    
    #extracting data for all variables on the required stations only
    print('Extracting variables ...')
    for u in range(len(sorted_indices)):
        print('station {} of {} ...'.format( u+1, len(sorted_indices)+1) )
        start = time.time()
        slf.readVariablesAtNode(sorted_indices[u])
        mresult = slf.getVarValuesAtNode()
        result[u , : , :] = mresult
        end = time.time()
        print('station {} is extracted'.format(station_names[u]))
        print('time elapsed: {} seconds'.format(end-start))
    
    #required variables from slf
    req_var = reqvar
    #importing telemac dictionary
    telemac_dict = teldict(telmod)
    
    #telemac and to be used names of the required variables
    telname = [telemac_dict[req_var[iu]][0] for iu in range(len(req_var))]
    usename = [telemac_dict[req_var[iu]][1] for iu in range(len(req_var))]
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    output_folder = basefol
    if not os.path.exists(path.join(output_folder , 'telemac_variables')):
        os.makedirs(path.join(output_folder , 'telemac_variables'))
    if not os.path.exists(path.join(output_folder, 'telemac_variables' , 'variables_all_stations')):
        os.makedirs(path.join(output_folder, 'telemac_variables' , 'variables_all_stations'))
        
    path_1 = path.join(output_folder , 'telemac_variables')
    path_1t = path.join(output_folder , 'telemac_variables' ,'variables_all_stations')
    
    print('{} is created'.format(path_1))
    print('{} is created'.format(path_1t))
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    #going through required variables   
    for v in range(len(req_var)):
        
        try:
            index_variable = variables.index(telname[v])
        except ValueError:
            print('Oops, a required variable is not in your selafin file ...')
            print(telname[v] + ' is not available')
            continue
        
        #make dir for each variable
        if not os.path.exists(path.join(path_1, usename[v])):
            os.makedirs(path.join(path_1, usename[v]))
        path_2 = path.join(path_1, usename[v])
        
        my_data_tot = np.empty((len(times) , len(station_names)+1) , dtype=np.object)
        #going through stations
        for n in range(len(station_names)):
            my_data = np.empty((len(times) , 2) , dtype=np.object)
            
            #going through timesteps
            for k in range(len(times)):
                my_data[k , 0] = date_array[k]
                my_data[k , 1] = np.transpose(result[n , k , index_variable])
                my_data_tot[k, 0] = date_array[k]
                my_data_tot[k, n+1] = np.transpose(result[n , k , index_variable])
            
            #converting numpy array to pandas dataframe
            my_data_df = pd.DataFrame(my_data , index = list(range(len(times))), columns = ['date'] + [usename[v]])
            #saving variables values for stations separately
            my_data_df.to_csv(path.join(path_2 , station_names[n] + '.dat' ))
            
        #onverting numpy array to pandas dataframe
        my_data_tot_df = pd.DataFrame(my_data_tot , index = list(range(len(times))), columns = ['date'] + stations)
        #saving variables values for all stations
        my_data_tot_df.to_csv(path.join(path_1t , usename[v] + '_all_stations'+ '.dat' ))
    
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    endt = time.time()
    print('variables are stored in the mentioned directories above')
    print('total time elapsed: {}'.format(datetime.timedelta(seconds = endt-startt)))
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    startuv = time.time()
    #merging velocity components and getting magnitudes vector and directions
    if 'VELOCITY U' and 'VELOCITY V' in variables:
        print('calculating velocities components...')
        print('magnitude and direction...')
        if not os.path.exists(path.join(output_folder , 'telemac_variables' , 'velocity_uv')):
            os.makedirs(path.join(output_folder , 'telemac_variables' , 'velocity_uv'))
        path_uv = path.join(output_folder , 'telemac_variables' ,'velocity_uv')
        
        suffix_list = ['magn' , 'dirc' , 'velu' ,'velv']
        stations_suffix = ['{}_{}'.format(a, b) for b in suffix_list for a in stations]
        data_uv_magn_dirc = pd.DataFrame(index=date_array , columns=stations_suffix)
        n=0
        for locn in station_names:
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            velocity_u = pd.read_csv(path.join(output_folder, 'telemac_variables', 'velocity_u' , locn + '.dat' ) , parse_dates = ['date'], 
                         date_parser = dateparse, index_col =0 , squeeze=True)
            velocity_u.set_index('date' , inplace = True)
            
            velocity_v = pd.read_csv(path.join(output_folder, 'telemac_variables', 'velocity_v' , locn + '.dat' ) , parse_dates = ['date'], 
                         date_parser = dateparse, index_col =0 , squeeze=True)
            velocity_v.set_index('date' , inplace = True)

            data_uv = pd.DataFrame(index=date_array , columns= [ 'magnitude' , 'direction' , 'velocity_v' , 'velocity_u'])
            for i in date_array:
                data_uv['magnitude'][i] = math.hypot(velocity_u.velocity_u[i] , velocity_v.velocity_v[i] )
                data_uv['direction'][i] = math.degrees(math.atan2(velocity_v.velocity_v[i] , velocity_u.velocity_u[i])) + 360*(velocity_v.velocity_v[i]<0) 
                data_uv['velocity_u'][i] = velocity_u.velocity_u[i]
                data_uv['velocity_v'][i] = velocity_v.velocity_v[i]
                
                data_uv_magn_dirc[locn+'_magn'][i] = math.hypot(velocity_u.velocity_u[i] , velocity_v.velocity_v[i] )
                data_uv_magn_dirc[locn+'_dirc'][i] = math.degrees(math.atan2(velocity_v.velocity_v[i] , velocity_u.velocity_u[i])) + 360*(velocity_v.velocity_v[i]<0)
                data_uv_magn_dirc[locn+'_velu'][i] = velocity_u.velocity_u[i]
                data_uv_magn_dirc[locn+'_velv'][i] = velocity_v.velocity_v[i]
            data_uv.to_csv(path.join(path_uv,locn+'.dat'))
            n+=1
            print('station {} of {} ...'.format( n, len(station_names)) )
            print(locn+' ...calculation done')
            print(locn+' ...merging done')
        data_uv_magn_dirc.to_csv(path.join(path_1t,'velocity_all_stations.dat'))
    enduv = time.time()
    print('total time elapsed: {}'.format(datetime.timedelta(seconds = enduv-startuv)))
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    startw = time.time()
    #merging wave components 
    if 'WAVE HEIGHT HM0' and 'MEAN DIRECTION' and 'MEAN PERIOD TM0' and 'PEAK PERIOD TPD' in variables:
        print('merging waves components... ... ...')
        
        suffix_list = ['swh' , 'mwd' , 'mwp' , 'pwp']
        stations_suffix = ['{}_{}'.format(a, b) for b in suffix_list for a in stations]
        data_wave = pd.DataFrame(index=date_array , columns=stations_suffix)
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        for locn in station_names:
            swh = pd.read_csv(path.join(output_folder, 'telemac_variables', 'wave_height' , locn + '.dat' ) , parse_dates = ['date'], 
                         date_parser = dateparse, index_col =0 , squeeze=True)
            swh.set_index('date' , inplace = True)
            
            mwd = pd.read_csv(path.join(output_folder, 'telemac_variables', 'wave_mean_direction' , locn + '.dat' ) , parse_dates = ['date'], 
                         date_parser = dateparse, index_col =0 , squeeze=True)
            mwd.set_index('date' , inplace = True)
            
            mwp = pd.read_csv(path.join(output_folder, 'telemac_variables', 'wave_mean_period_two' , locn + '.dat' ) , parse_dates = ['date'], 
                         date_parser = dateparse, index_col =0 , squeeze=True)
            mwp.set_index('date' , inplace = True)
            
            pwp = pd.read_csv(path.join(output_folder, 'telemac_variables', 'wave_peak_period' , locn + '.dat' ) , parse_dates = ['date'], 
                         date_parser = dateparse, index_col =0 , squeeze=True)
            pwp.set_index('date' , inplace = True)
            
            for i in date_array:
                data_wave[locn+'_swh'][i] = swh.wave_height[i]
                data_wave[locn+'_mwd'][i] = mwd.wave_mean_direction[i]
                data_wave[locn+'_mwp'][i] = mwp.wave_mean_period_two[i]
                data_wave[locn+'_pwp'][i] = pwp.wave_peak_period[i]
        data_wave.to_csv(path.join(path_1t ,'wave_all_stations.dat'))
        print('all stations ...merging done')
    endw = time.time()
    print('total time elapsed: {}'.format(datetime.timedelta(seconds = endw-startw)))
    