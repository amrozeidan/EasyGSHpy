#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:26:02 2019

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

def d_salinitycomp( commonfol , basefol , period_s , period_e  ,  requiredStationsFile ):
    
    station_names_req = np.loadtxt(requiredStationsFile , delimiter='\t', dtype = str).tolist()

    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_simul = pd.read_csv(path.join(basefol, 'telemac_variables','variables_all_stations' ,'salinity_all_stations.dat' ) , 
                           header =0 , parse_dates = ['date'],date_parser = dateparse, index_col =0 , squeeze=True)
    df_simul.set_index('date' , inplace = True)
    
    dateparse2 = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    path2 = path.join(commonfol, 'measurements')
    for file in os.listdir(path2):
        if file.endswith('.sa.dat'):
            df_meas = pd.read_csv( path.join(path2 , file) , 
                           header =0 , parse_dates = ['Time'],date_parser = dateparse2, index_col =0 , squeeze=True)
    
    #expanding the depths of salinity measurements into a second column index
    idx_depth = df_meas.columns.str.split('_', expand=True)
    df_meas.columns = idx_depth
    #adding a depth of 000 for simulated salinity values (keep it _simul)
    df_simul = df_simul.add_suffix('_simul')
    df_simul.columns = df_simul.columns.str.split('_', expand=True)
    
    station_names_for_comp=[]
    for station in station_names_req:
        if (station in df_meas.columns.levels[0]) and (station in df_simul.columns.levels[0]):
            print(station)
            station_names_for_comp.append(station)
        elif station not in df_meas.columns:
            print(station + ' does not have measured data')
        elif station not in df_simul.columns:
            print(station + ' does not have simulated data')
     
    df_meas = df_meas[station_names_for_comp][ period_s : period_e ]
    df_simul = df_simul[station_names_for_comp][ period_s : period_e ]
    

    #making directory to save the comparisons
    if not os.path.exists(path.join(basefol , 'salinitycomp')):
        os.makedirs(os.path.join(basefol , 'salinitycomp'))
    path_1 = path.join(basefol , 'salinitycomp')
    
    #creating a dataframe for salinity differences
    df_simul_meas = df_meas.copy(deep=True)
    
    for station in station_names_for_comp : 
        for i in df_meas[station].columns.to_list():
            df_simul_meas.loc[: , pd.IndexSlice[station , i]] = df_simul.loc[: , pd.IndexSlice[station , 'simul']]- df_meas.loc[: , pd.IndexSlice[station , i]]
     
    date_plots = df_simul_meas.index.to_list()
    [max_meas_simul , max_diff] = [ np.fmax( np.nanmax(df_meas.max()) , np.nanmax(df_simul.max()) ), np.nanmax(df_simul_meas.max())]
    [min_meas_simul , min_diff] = [ np.fmin( np.nanmin(df_meas.min()) , np.nanmin(df_simul.min()) ), np.nanmin(df_simul_meas.min())]
    
    #comparison plots
    for station in station_names_for_comp:
        for i in df_meas[station].columns.to_list():
            depth_for_plot = i
            #simulated values
            y2 = df_simul.loc[: , pd.IndexSlice[station , 'simul']].values
            #measured values for specific depth
            y1 = df_meas.loc[: , pd.IndexSlice[station , i]].values
            #difference values for specific depth
            y3 = df_simul_meas.loc[: , pd.IndexSlice[station , i]].values
            
            if df_meas.loc[: , pd.IndexSlice[station , i]].isnull().all() == False:
                #subplot1 meas vs simul, subplot2 diff; for each station (excluding nan)
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True , figsize=(15,10))
                
                ax1.plot( date_plots , y1 , 'tab:orange', date_plots , y2 , 'tab:blue')
                ax1.set_ylim(min_meas_simul-1 , max_meas_simul+1)
                ax1.set_title('Salinity comparison,station: ' + station + ', depth= ' + i)
                ax1.legend(['Measurements' , 'Simulations'])
                #ax1.set_xlabel('Date/Time')
                ax1.set_ylabel('Salinity')
                
#                df_simul_meas[station].plot( y = [i] , ax=ax2 , legend=True , 
#                                                  grid = True ,title = 'Salinity difference,station: ' + station + ', depth= ' + i,
#                                                  figsize = (15 , 10))
                ax2.plot(date_plots , y3)
                ax2.set_ylim(min_diff-1 , max_diff+1)
                ax2.set_title('Salinity difference,station: ' + station + ', depth= ' + i)  
                ax2.legend(['Difference'])
                ax2.set_xlabel('Date/Time')
                ax2.set_ylabel('Salinity difference')
                #plt.subplots_adjust(hspace=0.3)
                savingname = path.join(path_1 , 'salinity_comp_diffr_station_'+station + '_' + i + '.png')
                fig.savefig(savingname )
                plt.close() 
    