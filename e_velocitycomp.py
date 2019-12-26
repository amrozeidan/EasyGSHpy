#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:42:13 2019

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

def e_velocitycomp( commonfol , basefol , period_s , period_e  ,  requiredStationsFile ):
    
    station_names_req = np.loadtxt(requiredStationsFile , delimiter='\t', dtype = str).tolist()

    df_simul = pd.read_csv(path.join(basefol, 'telemac_variables','variables_all_stations' ,'velocity_all_stations.dat' ) ,index_col =0 )
    df_simul.index = pd.to_datetime(df_simul.index)
    
    dateparse2 = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    path2 = path.join(commonfol, 'measurements')
    for file in os.listdir(path2):
        if file.endswith('.cu.dat'):
            df_meas = pd.read_csv( path.join(path2 , file) , 
                           header =0 , parse_dates = ['Time'],date_parser = dateparse2, index_col =0 , squeeze=True)
    
    #expanding depths and velocity components
    idx_depth = df_meas.columns.str.split('_', expand=True)
    df_meas.columns = idx_depth
    #expanding velocity components 
    df_simul = df_simul.add_suffix('_simul')
    df_simul.columns = df_simul.columns.str.split('_', expand=True)

    #stations for comparison    
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
    if not os.path.exists(path.join(basefol , 'velocitycomp')):
        os.makedirs(os.path.join(basefol , 'velocitycomp'))
    path_1 = path.join(basefol , 'velocitycomp')
    
    #creating a dataframe for salinity differences
    df_simul_meas = df_meas.copy(deep=True)
    
    
    #i is representing depth
    #j is representing velocity component
    for station in station_names_for_comp : 
        for i,j in df_meas[station].columns.to_list():
            df_simul_meas.loc[: , pd.IndexSlice[station , i , j]] = df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]]- df_meas.loc[: , pd.IndexSlice[station , i , j]]
    
    #dataframe with no nan
    #df_for_nonan = pd.concat([df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]] , df_meas.loc[: , pd.IndexSlice[station , i , j]] ], axis = 1)
    
    #dates for plots
    date_plots = df_simul_meas.index.to_list()

    #max and min for plots scale
    [magn_max_meas_simul , magn_max_diff] = [ np.fmax( np.nanmax(df_meas.loc[: , pd.IndexSlice[: , : , 'magn']].max()) ,
                                                       np.nanmax(df_simul.loc[: , pd.IndexSlice[: , 'magn' , 'simul' ]].max()) ), 
                                                       np.nanmax(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'magn']] .max())]
    [dirc_max_meas_simul , dirc_max_diff] = [ np.fmax( np.nanmax(df_meas.loc[: , pd.IndexSlice[: , : , 'dirc']].max()) ,
                                                       np.nanmax(df_simul.loc[: , pd.IndexSlice[: , 'dirc' , 'simul' ]].max()) ), 
                                                       np.nanmax(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'dirc']] .max())]
    [velv_max_meas_simul , velv_max_diff] = [ np.fmax( np.nanmax(df_meas.loc[: , pd.IndexSlice[: , : , 'velv']].max()) ,
                                                       np.nanmax(df_simul.loc[: , pd.IndexSlice[: , 'velv' , 'simul' ]].max()) ), 
                                                       np.nanmax(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'velv']] .max())]
    [velu_max_meas_simul , velu_max_diff] = [ np.fmax( np.nanmax(df_meas.loc[: , pd.IndexSlice[: , : , 'velu']].max()) ,
                                                       np.nanmax(df_simul.loc[: , pd.IndexSlice[: , 'velu' , 'simul' ]].max()) ), 
                                                       np.nanmax(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'velu']] .max())]
    
    [magn_min_meas_simul , magn_min_diff] = [ np.fmin( np.nanmin(df_meas.loc[: , pd.IndexSlice[: , : , 'magn']].min()) ,
                                                       np.nanmin(df_simul.loc[: , pd.IndexSlice[: , 'magn' , 'simul' ]].min()) ), 
                                                       np.nanmin(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'magn']] .min())]
    [dirc_min_meas_simul , dirc_min_diff] = [ np.fmin( np.nanmin(df_meas.loc[: , pd.IndexSlice[: , : , 'dirc']].min()) ,
                                                       np.nanmin(df_simul.loc[: , pd.IndexSlice[: , 'dirc' , 'simul' ]].min()) ), 
                                                       np.nanmin(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'dirc']] .min())]
    [velv_min_meas_simul , velv_min_diff] = [ np.fmin( np.nanmin(df_meas.loc[: , pd.IndexSlice[: , : , 'velv']].min()) ,
                                                       np.nanmin(df_simul.loc[: , pd.IndexSlice[: , 'velv' , 'simul' ]].min()) ), 
                                                       np.nanmin(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'velv']] .min())]
    [velu_min_meas_simul , velu_min_diff] = [ np.fmin( np.nanmin(df_meas.loc[: , pd.IndexSlice[: , : , 'velu']].min()) ,
                                                       np.nanmin(df_simul.loc[: , pd.IndexSlice[: , 'velu' , 'simul' ]].min()) ), 
                                                       np.nanmin(df_simul_meas.loc[: , pd.IndexSlice[: , : , 'velu']] .min())]
    
    #dict for plots
    plot_dict = {'magn': 'Velocity Magnitude', 'dirc': 'Velocity Direction', 'velv': 'Velocity v' , 'velu': 'Velocity u'} 
    plot_dict_units = {'magn': '[m/s]', 'dirc': '[add units]', 'velv': '[m/s]' , 'velu': '[m/s]'} 
    
    plot_dict_min = {'magn': magn_min_meas_simul, 'dirc': dirc_min_meas_simul, 'velv': velv_min_meas_simul , 'velu': velu_min_meas_simul} 
    plot_dict_max = {'magn': magn_max_meas_simul, 'dirc': dirc_max_meas_simul, 'velv': velv_max_meas_simul , 'velu': velu_max_meas_simul} 
    
    plot_dict_min_diff = {'magn': magn_min_diff, 'dirc': dirc_min_diff, 'velv': velv_min_diff , 'velu': velu_min_diff} 
    plot_dict_max_diff = {'magn': magn_max_diff, 'dirc': dirc_max_diff, 'velv': velv_max_diff , 'velu': velu_max_diff} 
    
    #comparison plots
    for station in station_names_for_comp:
        for i,j in df_meas[station].columns.to_list():
                        
            #simulated values
            y2 = df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]].values
            #measured values for specific depth
            y1 = df_meas.loc[: , pd.IndexSlice[station , i , j]].values
            #difference values for specific depth
            y3 = df_simul_meas.loc[: , pd.IndexSlice[station , i , j]].values
            
            if df_meas.loc[: , pd.IndexSlice[station , i , j]].isnull().all() == False:
                #subplot1 meas vs simul, subplot2 diff; for each station (excluding nan)
                
                df_for_calc = pd.concat([df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]] , df_meas.loc[: , pd.IndexSlice[station , i , j]] ], axis = 1) 
                x = df_for_calc.iloc[1]
                y = df_for_calc.iloc[0]
                mse = mean_squared_error(x,y)
                rmse = mse**0.5 
                me = np.mean(y-x)
                mae = mean_absolute_error(x,y)
                
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True , figsize=(15,10))
                
                ax1.plot( date_plots , y1 , 'tab:orange', date_plots , y2 , 'tab:blue')
                ax1.set_ylim(plot_dict_min[j]-1 , plot_dict_max[j]+1)
                ax1.set_title( plot_dict[j] +' comparison,station: ' + station + ', depth= ' + i)
                ax1.legend(['Measurements' , 'Simulations'])
                ax1.set_xlabel('Date/Time')
                ax1.set_ylabel(plot_dict[j] + ' ' + plot_dict_units[j])
                
                
#                df_simul_meas.loc[: , pd.IndexSlice[station , i ,j]].plot(  ax=ax2 , legend=True , 
#                                                  grid = True ,title = plot_dict[j] + ' difference,station: ' + station + ', depth= ' + i,
#                                                  figsize = (15 , 10))
                
                ax2.plot(date_plots , y3)
                ax2.set_ylim(plot_dict_max_diff[j]-1 ,plot_dict_min_diff[j]+1)
                ax2.legend(['Difference'])
                ax2.set_title(plot_dict[j] + ' difference,station: ' + station + ', depth= ' + i)
                ax2.set_xlabel('Date/Time')
                ax2.set_ylabel(plot_dict[j] + ' difference' + ' '+ plot_dict_units[j])
                anchored_text = AnchoredText('RMSE='+str(round(rmse,5))+'\nMEA='+str(round(mae,5))+'\nME='+str(round(me,5)), loc=4)
                ax2.add_artist(anchored_text)
                
                plt.subplots_adjust(hspace=0.3)
                savingname = path.join(path_1 , plot_dict[j] + '_comp_diffr_station_'+station + '_' + i + '.png')
                fig.savefig(savingname )
                plt.close()  