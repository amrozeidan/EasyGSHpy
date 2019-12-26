#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:53:19 2019

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
from datetime import datetime as dt
import utide


commonfol = '/Users/amrozeidan/Documents/hiwi/easygshpy/com'
basefol = '/Users/amrozeidan/Documents/hiwi/easygshpy/base'
period_s = '2015-01-06'  
period_e = '2015-01-12'
k = 147
offset = 0
requiredStationsFile = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/required_stations.dat'
stationsDB = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/info_all_stations.dat'

def c_excoef( commonfol , basefol , period_s , period_e  , requiredStationsFile , stationsDB ):
    
    #required stations
    station_names_req = np.loadtxt(requiredStationsFile , delimiter='\t', dtype = str).tolist()

    #extracting data from all stations database
    maindatabase = pd.read_csv(stationsDB , header = 0 , delimiter = ',')
    maindatabase.set_index('name' , inplace = True) 
        
    #import simulated values
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_simul = pd.read_csv(path.join(basefol, 'telemac_variables','variables_all_stations' ,'free_surface_all_stations.dat' ) , 
                           header =0 , parse_dates = ['date'],date_parser = dateparse, index_col =0 , squeeze=True)
    df_simul.set_index('date' , inplace = True)
    
    #import measured values 
    dateparse2 = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    path2 = path.join(commonfol, 'measurements')
    for file in os.listdir(path2):
        if file.endswith('.wl.dat'):
            df_meas = pd.read_csv( path.join(path2 , file) , 
                           header =0 , parse_dates = ['Time'],date_parser = dateparse2, index_col =0 , squeeze=True)
    
    #sations to be extracted
    station_names_for_comp=[]
    for station in station_names_req:
        if (station in df_meas.columns) and (station in df_simul.columns) and (station in maindatabase.index):
            print(station)
            station_names_for_comp.append(station)
#        elif station not in df_meas.columns:
#            print(station + ' does not have measured data')
#        elif station not in df_simul.columns:
#            print(station + ' does not have simulated data')
#        elif station not in stations:
#            print(station + ' does not have enough data in StationsDatabase')
    
    #slicing the required stations and adding _meas and _simul to suffixes
    # to avoid duplication (_x and _y) while joining 
    dfmeasforcomp =   df_meas[station_names_for_comp][period_s : period_e ]
    dfmeasforcomp = dfmeasforcomp.add_suffix('_meas')
    dfsimulforcomp = df_simul[station_names_for_comp][period_s : period_e ]
    dfsimulforcomp = dfsimulforcomp.add_suffix('_simul')
    dfmeassimulforcomp = dfmeasforcomp.join(dfsimulforcomp, how = 'inner')
    dfmeassimulforcomp = dfmeassimulforcomp.sort_index(axis = 1)
    #adding a second level in suffixes (multiindexing)
    #first level station names, second level meas, simul
    dfmeassimulforcomp.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in dfmeassimulforcomp.columns])
    #keep the order of stations
    dfmeassimulforcomp = dfmeassimulforcomp[station_names_for_comp]

    #converting datetime to datenum
    def datenumaz(d):
        return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)
            
    #and changing the index to datenum
    dfmeassimulforcomp.index = dfmeassimulforcomp.index.map(datenumaz)
    
    datenum = dfmeassimulforcomp.index

    #required tides avreviations
    pTides = ['MM','MF','Q1','O1','K1','SO1','MU2','N2','NU2','M2','S2','2SM2','MO3','MN4','M4'
          ,'MS4','MK4','S4','M6','2MS6','S6','M8','M10','M12' ] 
   
    #making directories for coefficients
    if not os.path.exists(os.path.join(basefol , 'coef_simulated')):
        os.makedirs(os.path.join(basefol , 'coef_simulated'))
    path_s = os.path.join(basefol , 'coef_simulated')
    
    if not os.path.exists(os.path.join(commonfol , 'coef_measured')):
        os.makedirs(os.path.join(commonfol , 'coef_measured'))
    path_m = os.path.join(commonfol , 'coef_measured')
    
    #preparing dataframes
    dfa = pd.DataFrame()
    dfg = pd.DataFrame()
    idx = pd.IndexSlice
    df_recons_h_meas = pd.DataFrame(columns = station_names_for_comp , index = datenum)
    df_recons_h_simul = pd.DataFrame(columns = station_names_for_comp , index = datenum)
    station_no_nan = []
    
    #extracting coef using utide.solve
    #utide.reconstruct to be revised, getting weird water levels
    for station in station_names_for_comp:
        
        latitude = maindatabase.Latitude[station]
        
        if dfmeassimulforcomp.loc[:,idx[station ,['meas']]].isnull().all().bool() == False:
            #MonteCarlo , ols
            #method = 'ols' , conf_int = 'MC'
            coef_meas = utide.solve(np.array(datenum) , dfmeassimulforcomp.loc[:,idx[station ,['meas']]].values[:,0] , lat = latitude , constit = pTides)
            #couldn't find how to save a coef
            #so i decided to merge the get_peaks function into this script
            coef_simul = utide.solve(np.array(datenum) , dfmeassimulforcomp.loc[:,idx[station ,['simul']]].values[:,0] , lat = latitude , constit = pTides)
            
            #reconstructing the coef for the peak comparison 
            recons_coef_meas = utide.reconstruct(np.array(datenum) , coef_meas)
            df_recons_h_meas[station] = recons_coef_meas['h']

            recons_coef_simul = utide.reconstruct(np.array(datenum) , coef_simul)
            df_recons_h_simul[station] = recons_coef_simul['h']

            measindex = coef_meas['name'].tolist()

            simulindex = coef_simul['name'].tolist()
            
            tempdfameas = pd.Series(coef_meas['A']).to_frame()
            tempdfameas.index = measindex
            tempdfameas.columns = [station] 
            tempdfameas.columns = pd.MultiIndex.from_product([tempdfameas.columns, ['meas']])

            tempdfasimul = pd.Series(coef_simul['A']).to_frame()
            tempdfasimul.index = simulindex 
            tempdfasimul.columns = [station] 
            tempdfasimul.columns = pd.MultiIndex.from_product([tempdfasimul.columns, ['simul']])
            
            dfa = pd.concat([dfa , tempdfameas] , axis = 1, sort = True)
            dfa = pd.concat([dfa , tempdfasimul] , axis = 1, sort = True)
            
            tempdfgmeas = pd.Series(coef_meas['g']).to_frame()
            tempdfgmeas.index = measindex
            tempdfgmeas.columns = [station] 
            tempdfgmeas.columns = pd.MultiIndex.from_product([tempdfgmeas.columns, ['meas']])

            tempdfgsimul = pd.Series(coef_simul['g']).to_frame()
            tempdfgsimul.index = simulindex    
            tempdfgsimul.columns = [station] 
            tempdfgsimul.columns = pd.MultiIndex.from_product([tempdfgsimul.columns, ['simul']])
            
            dfg = pd.concat([dfg , tempdfgmeas] , axis = 1, sort = True)
            dfg = pd.concat([dfg , tempdfgsimul] , axis = 1, sort = True)
            
            station_no_nan.append(station)
    
    #save amplitude and phase shift for all stations
    dfa.loc[:,idx[station_no_nan ,['meas']]].to_csv(path.join(path_m,'meas_amplitude_all_stations.dat'))
    dfa.loc[:,idx[station_no_nan ,['simul']]].to_csv(path.join(path_s,'simul_amplitude_all_stations.dat'))
    dfg.loc[:,idx[station_no_nan ,['meas']]].to_csv(path.join(path_m,'meas_phaseshift_all_stations.dat'))
    dfg.loc[:,idx[station_no_nan ,['simul']]].to_csv(path.join(path_s,'simul_phaseshift_all_stations.dat'))
    
    #adding diff column for each station
    #A
    a = dfa.loc[:,pd.IndexSlice[:,'simul']].sub(dfa.loc[:,pd.IndexSlice[:,'meas']].values, 1).rename(columns={'simul':'diffr'})
    dfaf = dfa.join(a).sort_index(axis=1)
    #keep the order of stations
    dfaf = dfaf[station_no_nan]
    #g
    a = dfg.loc[:,pd.IndexSlice[:,'simul']].sub(dfg.loc[:,pd.IndexSlice[:,'meas']].values, 1).rename(columns={'simul':'diffr'})
    dfgf = dfg.join(a).sort_index(axis=1)
    #keep the order of stations
    dfgf = dfgf[station_no_nan]
    
    #all data dataframe
    dfag = pd.concat([dfaf , dfgf] , keys = ['A' , 'g'])   

    #plots 
    #making directory to save the comparisons
    if not os.path.exists(os.path.join(basefol , 'ptcomp')):
        os.makedirs(os.path.join(basefol , 'ptcomp'))
    path_1 = os.path.join(basefol , 'ptcomp')
    
    #transpose df
    dfaft = dfaf.T
    dfgft = dfgf.T

    #looping through tides
    for tide in pTides:
        
        #A
        #subplot1 meas vs simul, subplot2 diff; for each tide
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False,figsize=(32,22))
        
        dfaft[tide].unstack(level=-1).reindex(station_no_nan).plot(y = ['meas' , 'simul'] ,  ax=ax1 , legend=True , 
                                              grid = True ,title = ' Amplitude comparison for tide: ' + tide, 
                                              figsize = (16 , 11) , style = 'x')
        dfaft[tide].unstack(level=-1).reindex(station_no_nan).plot( y = 'diffr' , ax=ax2 , legend=True , 
                                              grid = True ,title = ' Amplitude difference for tide: ' + tide,
                                              figsize = (16 , 11), style = 'x')
        ax1.set_ylabel('Amplitude [m]')
        ax1.legend(['Measured' , 'Simulated'])
        ax1.set_xticks(list(range(len(station_no_nan))))
        ax1.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
        plt.subplots_adjust(hspace=0.5)
        ax2.set_ylabel('Amplitude Difference [m]')
        ax2.set_xlabel('Stations')
        ax2.legend(['Difference'])
        ax2.set_xticks(list(range(len(station_no_nan))))
        ax2.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
        
        savingname = path.join(path_1 , tide + '_amplitude_comp' + '.png')
        fig.savefig(savingname)
        plt.close()
    
        #g
        #subplot1 meas vs simul, subplot2 diff; for each tide
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False,figsize=(32,22))
        
        dfgft[tide].unstack(level=-1).reindex(station_no_nan).plot(y = ['meas' , 'simul'] ,  ax=ax1 , legend=True , 
                                              grid = True ,title = ' Phase shift comparison for tide: ' + tide, 
                                              figsize = (16 , 11) , style = 'x')
        dfgft[tide].unstack(level=-1).reindex(station_no_nan).plot( y = 'diffr' , ax=ax2 , legend=True , 
                                              grid = True ,title = ' Phase shift difference for tide: ' + tide,
                                              figsize = (16 , 11), style = 'x')
        ax1.set_ylabel('Phase shift []')
        ax1.set_xticks(list(range(len(station_no_nan))))
        ax1.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
        ax1.legend(['Measured' , 'Simulated'])
        plt.subplots_adjust(hspace=0.5)
        ax2.set_ylabel('Phase shift Difference []')
        ax2.set_xlabel('Stations')
        ax2.set_xticks(list(range(len(station_no_nan))))
        ax2.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
        ax2.legend(['Difference'])
        
        savingname = path.join(path_1 , tide + '_phaseshift_comp' + '.png')
        fig.savefig(savingname)
        plt.close()
        
#add get peaks function
#plt.plot(hmeas) 
#plt.plot(loc_meas, hmeas[loc_meas], "x")   
