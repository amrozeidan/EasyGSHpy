#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:21:00 2019

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

commonfol = '/Users/amrozeidan/Documents/hiwi/easygshpy/com'
basefol = '/Users/amrozeidan/Documents/hiwi/easygshpy/base'
period_s = '2015-01-06'  
period_e = '2015-01-12'
#k = 147
offset = 0
requiredStationsFile = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/required_stations.dat'

#shrink the multi-indices into one level
#df_img.columns = ['_'.join(col).strip() for col in df_img.columns.values]

def f_wavecomp( commonfol , basefol , period , offset ,  requiredStationsFile ):

    station_names_req = np.loadtxt(requiredStationsFile , delimiter='\t', dtype = str).tolist()

#    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_simul = pd.read_csv(path.join(basefol, 'telemac_variables','variables_all_stations' ,'wave_all_stations.dat' ) ,index_col =0 )
    df_simul.index = pd.to_datetime(df_simul.index)
#                           header =0 , parse_dates = ['Unnamed: 0'],date_parser = dateparse, index_col =0 , squeeze=True)
#    df_simul.set_index('Unnamed: 0', inplace = True)
    
    dateparse2 = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    path2 = path.join(commonfol, 'measurements')
    for file in os.listdir(path2):
        if file.endswith('.wv.dat'):
#            df_meas = pd.read_csv( path.join(path2 , file) , 
#                           header =0 , parse_dates = ['Time'],date_parser = dateparse2, index_col =0 , squeeze=True)
            ####for testing fake data
            df_meas = pd.read_csv( path.join(path2 , file) , index_col = 0)

    #expanding wave components
    idx = df_meas.columns.str.split('_', expand=True)
    df_meas.columns = idx
    idx = df_simul.columns.str.split('_', expand=True)
    df_simul.columns = idx
    
    
    station_names_for_comp=[]
    for station in station_names_req:
        if (station in df_meas.columns.levels[0]) and (station in df_simul.columns.levels[0]):
            print(station)
            station_names_for_comp.append(station)
        elif station not in df_meas.columns:
            print(station + ' does not have measured data')
        elif station not in df_simul.columns:
            print(station + ' does not have simulated data')
            
    ####df_meas['Antifer'] = np.nan
     
    df_meas = df_meas[station_names_for_comp][ period_s : period_e ]
    df_simul = df_simul[station_names_for_comp][ period_s : period_e ]
    
    #join headers again 
    df_meas.columns = ['_'.join(col).strip() for col in df_meas.columns.values]
    df_simul.columns = ['_'.join(col).strip() for col in df_simul.columns.values]
    
    dfmeasforcomp = df_meas.add_suffix('_meas')
    dfsimulforcomp = df_simul.add_suffix('_simul')
    
    dfmeassimulforcomp = dfmeasforcomp.join(dfsimulforcomp, how = 'inner')
    dfmeassimulforcomp = dfmeassimulforcomp.sort_index(axis = 1)
    
    #adding a second level in suffixes (multi-indexing)
    #first level station names, second level meas, simul and diff
    dfmeassimulforcomp.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in dfmeassimulforcomp.columns])
    #adding diff column for each station
    a = dfmeassimulforcomp.loc[:,pd.IndexSlice[:,:,'simul']].sub(dfmeassimulforcomp.loc[:,pd.IndexSlice[:,:,'meas']].values, 1).rename(columns={'simul':'diffr'})
    dfmeassimulforcomp = dfmeassimulforcomp.join(a).sort_index(axis=1)
    #keep the order of stations
    dfmeassimulforcomp = dfmeassimulforcomp[station_names_for_comp]
    
    #max and min (y-axis) for plots scales
    [swh_max_meas_simul , swh_max_diff] = [ np.fmax(   np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'swh' , :]].max()) ,
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'swh' , :]].max()) ), 
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'swh' , 'diffr']] .max())]
    [mwd_max_meas_simul , mwd_max_diff] = [ np.fmax(   np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwd' , :]].max()) ,
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwd' , :]].max()) ), 
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwd' , 'diffr']] .max())]
    [mwp_max_meas_simul , mwp_max_diff] = [ np.fmax( np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwp' , :]].max()) ,
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwp' , :]].max()) ), 
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwp' , 'diffr']] .max())]
    [pwp_max_meas_simul , pwp_max_diff] = [ np.fmax( np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'pwp' , :]].max()) ,
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'pwp' , :]].max()) ), 
                                                       np.nanmax(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'pwp' , 'diffr']] .max())]
    
    [swh_min_meas_simul , swh_min_diff] = [ np.fmin( np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'swh' , :]].min()) ,
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'swh' , :]].min()) ), 
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'swh' , 'diffr']] .min())]
    [mwd_min_meas_simul , mwd_min_diff] = [ np.fmin( np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwd' , :]].min()) ,
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwd' , :]].min()) ), 
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwd' , 'diffr']] .min())]
    [mwp_min_meas_simul , mwp_min_diff] = [ np.fmin( np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwp' , :]].min()) ,
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwp' , :]].min()) ), 
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'mwp' , 'diffr']] .min())]
    [pwp_min_meas_simul , pwp_min_diff] = [ np.fmin( np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'pwp' , :]].min()) ,
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'pwp' , :]].min()) ), 
                                                       np.nanmin(dfmeassimulforcomp.loc[: , pd.IndexSlice[: , 'pwp' , 'diffr']] .min())]
    
    #dict for plots
    plot_dict_min = {'swh': swh_min_meas_simul, 'mwd': mwd_min_meas_simul, 'mwp': mwp_min_meas_simul , 'pwp': pwp_min_meas_simul} 
    plot_dict_max = {'swh': swh_max_meas_simul, 'mwd': mwd_max_meas_simul, 'mwp': mwp_max_meas_simul , 'pwp': pwp_max_meas_simul} 
    
    plot_dict_min_diff = {'swh': swh_min_diff, 'mwd': mwd_min_diff, 'mwp': mwp_min_diff , 'pwp': pwp_min_diff} 
    plot_dict_max_diff = {'swh': swh_max_diff, 'mwd': mwd_max_diff, 'mwp': mwp_max_diff , 'pwp': pwp_max_diff} 
    
    #making directory to save the comparisons
    if not os.path.exists(path.join(basefol , 'wavecomp')):
        os.makedirs(os.path.join(basefol , 'wavecomp'))
    path_1 = path.join(basefol , 'wavecomp')
    
    #dict for plots
    plot_dict = {'swh': 'Wave height', 'mwd': 'Wave direction', 'mwp': 'Wave mean period' , 'pwp': 'Wave peak period'} 
    plot_dict_units = {'swh': '[add units]', 'mwd': '[add units]', 'mwp': '[add units]' , 'pwp': '[add units]'} 
    
    #sort columns multiindices to avoid 'PerformanceWarning: indexing past lexsort depth may impact performance.'
    dfmeassimulforcomp = dfmeassimulforcomp.sort_index( axis = 1)
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    print('Extracting wave components comparison plots ...')
    
    n=0
    #comparison plots
    for station in station_names_for_comp:
        
        n+=1
        print('station {} of {} ...'.format( n, len(station_names_for_comp)) )
        
        for i in ['swh' , 'mwd' , 'mwp' , 'pwp']:
            
            try:
                #as some of the components are not included in the measured values
                dfmeassimulforcomp.xs((station , i) , axis =1)
            except:
                print( i + 'is not available for station:' + station)
                continue
            
            if dfmeassimulforcomp.loc[: , pd.IndexSlice[station , i , 'meas']].isnull().all() == False:

                #subplot1 meas vs simul, subplot2 diff; for each station (including nan)
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False , figsize=(30,20))
                    
#                dfmeassimulforcomp.loc[: , pd.IndexSlice[station , i]].plot( y = ['meas' , 'simul'] , ax=ax1 , legend=True  , 
#                                                  ylim = (min(minmeas,minsimul)-1 , max(maxmeas,maxsimul)+1),
#                                                  grid = True ,title = plot_dict[i]+' comparison,Station: '+station , 
#                                                  figsize = (15 , 10))
                dfmeassimulforcomp.xs((station , i) , axis =1).plot( y = ['meas' , 'simul'] , ax=ax1 , legend=True  , 
                                                  ylim = (plot_dict_min[i]-1 , plot_dict_max[i]+1),
                                                  grid = True ,title = plot_dict[i]+' comparison,Station: '+station , 
                                                  figsize = (15 , 10))
                ax1.legend(['Measurements' , 'Simulations'])
                ax1.set_ylabel(plot_dict[i] + ' ' + plot_dict_units[i])   

                dfmeassimulforcomp.xs((station , i) , axis =1).plot( y = ['diffr'] , ax=ax2 , legend=True , 
                                                  ylim = (plot_dict_min_diff[i] -1 , plot_dict_max_diff[i] +1),
                                                  grid = True ,title = plot_dict[i] + ' difference, Station: '+station ,
                                                  figsize = (15 , 10))
                ax2.legend(['Differences' ])
                ax2.set_ylabel(plot_dict[i] + ' difference' + plot_dict_units[i])
                ax2.set_xlabel('Date/Time[UTC]')
                
                plt.subplots_adjust(hspace=0.3)
                
                savingname = path.join(path_1 , plot_dict[i] + '_comp_diffr_station_'+station + '.png')
                fig.savefig(savingname )
                plt.close()
                
        print(station+' ...wave components comparison extracted ...')
        print('-------------------------------------')
                        
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')