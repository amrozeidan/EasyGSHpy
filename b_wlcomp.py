#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:03:23 2019

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
k = 147
offset = 0
requiredStationsFile = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/required_stations.dat'

def b_wlcomp( commonfol , basefol , period_s , period_e , k , requiredStationsFile ):
    
    station_names_req = np.loadtxt(requiredStationsFile , delimiter='\t', dtype = str).tolist()

    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_simul = pd.read_csv(path.join(basefol, 'telemac_variables','variables_all_stations' ,'free_surface_all_stations.dat' ) , 
                           header =0 , parse_dates = ['date'],date_parser = dateparse, index_col =0 , squeeze=True)
    df_simul.set_index('date' , inplace = True)
    
    dateparse2 = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    path2 = path.join(commonfol, 'measurements')
    for file in os.listdir(path2):
        if file.endswith('.wl.dat'):
            df_meas = pd.read_csv( path.join(path2 , file) , 
                           header =0 , parse_dates = ['Time'],date_parser = dateparse2, index_col =0 , squeeze=True)
    
    station_names_for_comp=[]
    for station in station_names_req:
        if (station in df_meas.columns) and (station in df_simul.columns):
            print(station)
            station_names_for_comp.append(station)
        elif station not in df_meas.columns:
            print(station + ' does not have measured data')
        elif station not in df_simul.columns:
            print(station + ' does not have simulated data')
            
    #comparison of wl and wl difference
    #making directory to save the comparisons
    if not os.path.exists(path.join(basefol , 'wlcomp')):
        os.makedirs(os.path.join(basefol , 'wlcomp'))
    path_1 = path.join(basefol , 'wlcomp')
    
    #slicing the required stations and adding _meas and _simul to suffixes
    # to avoid duplication (_x and _y) while joining 
    dfmeasforcomp =   df_meas[station_names_for_comp] 
    dfmeasforcomp = dfmeasforcomp.add_suffix('_meas')
    dfsimulforcomp = df_simul[station_names_for_comp]
    dfsimulforcomp = dfsimulforcomp.add_suffix('_simul')
    
    dfmeassimulforcomp = dfmeasforcomp.join(dfsimulforcomp, how = 'inner')
    dfmeassimulforcomp = dfmeassimulforcomp.sort_index(axis = 1)
    
    #slice to the needed period
    dfmeassimulforcomp[ period_s : period_e ]
    
    #adding a second level in suffixes (multi-indexing)
    #first level station names, second level meas, simul and diff
    dfmeassimulforcomp.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in dfmeassimulforcomp.columns])
    #adding diff column for each station
    a = dfmeassimulforcomp.loc[:,pd.IndexSlice[:,'simul']].sub(dfmeassimulforcomp.loc[:,pd.IndexSlice[:,'meas']].values, 1).rename(columns={'simul':'diffr'})
    dfmeassimulforcomp = dfmeassimulforcomp.join(a).sort_index(axis=1)
    #keep the order of stations
    dfmeassimulforcomp = dfmeassimulforcomp[station_names_for_comp]
    
    #defining moving average
    def movingavg(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    #extracting moving average of water level difference
    dfwldiffrmovavrg = dfmeassimulforcomp.loc[:,pd.IndexSlice[:,'diffr']].apply( lambda x : list(movingavg(x,k))).rename(columns={'diffr':'movavrg'})
    dfwldiffrmovavrg.index = dfmeassimulforcomp.index
    #saving moving average dataframe
    dfwldiffrmovavrg.to_csv(path.join(path_1,'WL_Diff_Moving_Avg_all_stations.dat' ))
    
    #extracting wl difference
    dfwldiff = dfmeassimulforcomp.filter(regex='diffr')
    dfwldiff.to_csv(path.join(path_1,'WL_Diff_all_stations.dat' ))
    
    #add tne moving average to the main dataframe 
    dfmeassimulforcomp = dfmeassimulforcomp.join(dfwldiffrmovavrg, how = 'inner')
    dfmeassimulforcomp = dfmeassimulforcomp.sort_index(axis = 1)
    
    #finding min and max for meas and simul, for all stations, used for plot scales later
    s = pd.Series(dfmeassimulforcomp.max())
    [maxmeas , maxsimul , maxdiffr] = [s.max(level=1).meas , s.max(level=1).simul , s.max(level=1).diffr]
    #finding min and max for diff, for all stations
    ss = pd.Series(dfmeassimulforcomp.min())
    [minmeas , minsimul , mindiffr] = [ss.min(level=1).meas , ss.min(level=1).simul , ss.min(level=1).diffr]
    
    #plots 
    #definitions for scatter plots below
    df = dfmeassimulforcomp
    idx = pd.IndexSlice
    totnrmse = []
    totrmse = []
    totmae = []
    totme = []
    station_no_nan = []
    
    #looping through required stations
    for station in station_names_for_comp:
        #excluding nan 
        dfstation = df.loc[idx[:] , idx[ station , ['meas' , 'simul']]] 
        dfstation = dfstation.dropna() #removing nan
        x =  dfstation[station].meas
        y =  dfstation[station].simul
        
        #getting RMSE , ME , MAE
        if x.empty==False and y.empty==False:
            mse = mean_squared_error(x,y)
            rmse = mse**0.5 
            me = np.mean(y-x)
            mae = mean_absolute_error(x,y)
        
            #subplot1 meas vs simul, subplot2 diff; for each station (including nan)
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False , figsize=(30,20))
            
            dfmeassimulforcomp[station].plot( y = ['meas' , 'simul', 'diffr'] , ax=ax1 , legend=True  , 
                                          ylim = (min(minmeas,minsimul)-1 , max(maxmeas,maxsimul)+1),
                                                  grid = True ,title = 'Water Level comparison,Station: '+station , 
                                                  figsize = (15 , 10))
            ax1.legend(['Measurements' , 'Simulations' , 'Differences'])
            
            anchored_text = AnchoredText('RMSE='+str(round(rmse,5))+'\nMEA='+str(round(mae,5))+'\nME='+str(round(me,5)), loc=4)
            ax1.add_artist(anchored_text)
            dfmeassimulforcomp[station].plot( y = ['diffr' , 'movavrg'] , ax=ax2 , legend=True , 
                                          ylim = (mindiffr -1 , maxdiffr +1),
                                                  grid = True ,title = 'Water Level difference, Station: '+station ,
                                                  figsize = (15 , 10))
            ax2.legend(['Differences' , 'Moving Average'])
            ax1.set_ylabel('Water level [m+NHN]/Differences [m]')
            ax2.set_ylabel('Water level difference [m]')
            ax2.set_xlabel('Date/Time[UTC]')
            #plt.subplots_adjust(hspace=0.5)
            savingname = path.join(path_1 , 'Wl_comp_diffr_station_'+station + '.png')
            fig.savefig(savingname )
            plt.close()
    
         
            #scatter plot of meas and simul data colored by density (excluding nan) 
            xy = np.vstack([ x, y ]) 
            z =   gaussian_kde(xy)(xy) 
                 
            fig, ax = plt.subplots()
            cax = ax.scatter( x , y , c=z, s=1, edgecolor='' , cmap=plt.cm.jet )
            
            line = mlines.Line2D([0, 1], [0, 1], color='black')
            transform = ax.transAxes
            line.set_transform(transform)
            ax.add_line(line)
            
            ax.set_title('Density Scatter Plot of Station: '+station)
            ax.set_xlabel('Measured Water Level')
            ax.set_ylabel('Simulated Water Level')
            cbar = fig.colorbar(cax)
            cbar.ax.set_title('Density')
            savingname = path.join(path_1 , station + '_Wl_scatter' + '.png')
            fig.savefig(savingname )
            plt.close()
            
            #nrmse extracting and storing
            nrmse = rmse/(x.max()-x.min())        
            totnrmse.append(nrmse) 
            totrmse.append(rmse)
            totmae.append(mae)
            totme.append(me)
            
            station_no_nan.append(station)
            
        else:
            print(station + ' has no measured values (all are nan). It will not be included in plots and errors calcs')
        
    #nrmse dataframe 
    d = pd.DataFrame(  totnrmse , index = station_no_nan , columns = ['NRMSE'] )
    #and  plot
    ax = d.plot(style='x' , grid = True , figsize = (20 , 12.5) ,
                title = 'NRMSE of WL on the required stations' )        
    ax.set_xlabel('Station Names')  
    ax.set_ylabel('NRMSE') 
    ax.set_xticks(list(range(len(station_no_nan))))
    ax.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
    fig = ax.get_figure()
    ax.set_ylabel('NRMSE [m]')
    ax.set_xlabel('Stations')
    savingname = path.join(path_1 , 'NRMSE of WL on the required locations' + '.png')
    fig.savefig(savingname  , dpi = 'figure' , orientation = 'landscape')
    plt.close()
    
    #rmse,mae and me dataframe 
    d = pd.DataFrame(  totrmse , index = station_no_nan , columns = ['RMSE'] )
    d['MAE'] = pd.Series(totmae , index=d.index)
    d1 = pd.DataFrame(  totme , index = station_no_nan , columns = ['ME'] )
    #and plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False , figsize=(32,22))
    
    d.plot(style='x' , grid = True , figsize = (16 , 11) ,ax=ax1 , legend=True ,
                title = 'Root Mean Square and Mean Absolute Error of Water Level' )  
    ax1.set_xticks(list(range(len(station_no_nan))))
    ax1.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
    
    d1.plot(style='x' , grid = True , figsize = (16 , 11) ,ax=ax2 , legend=True ,
                title = 'Mean Error of Water Level' )        
    ax2.set_xlabel('Station Names')
    ax2.set_xticks(list(range(len(station_no_nan))))
    ax2.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
    
    ax1.set_ylabel('RMSE,MAE [m]')
    ax2.set_ylabel('ME [m]')
    ax2.set_xlabel('Stations')
    plt.subplots_adjust(hspace=0.5)
    savingname = path.join(path_1 , 'RMSE of WL on the required locations' + '.png')
    fig.savefig(savingname  , dpi = 'figure' , orientation = 'landscape')
    plt.close()
    
    