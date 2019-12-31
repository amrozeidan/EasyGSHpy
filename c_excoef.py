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
from scipy.spatial.distance import cdist 


commonfol = '/Users/amrozeidan/Documents/hiwi/easygshpy/com2612'
basefol = '/Users/amrozeidan/Documents/hiwi/easygshpy/base2612'
period_s = '2015-01-06'  
period_e = '2015-01-12'
k = 147
offset = 0
requiredStationsFile = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/required_stations.dat'
stationsDB = '/Users/amrozeidan/Documents/hiwi/easygshpy/stationsDB_reqStations/info_all_stations.dat'

def findpeaks(series, DELTA):
    """
    Finds extrema in a pandas series data.

    Parameters
    ----------
    series : `pandas.Series`
        The data series from which we need to find extrema.

    DELTA : `float`
        The minimum difference between data values that defines a peak.

    Returns
    -------
    minpeaks, maxpeaks : `list`
        Lists consisting of pos, val pairs for both local minima points and
        local maxima points.
    """
    # Set inital values
    mn, mx = np.Inf, -np.Inf
    minpeaks = []
    maxpeaks = []
    lookformax = True
    start = True
    # Iterate over items in series
    for time_pos, value in series.iteritems():
        if value > mx:
            mx = value
            mxpos = time_pos
        if value < mn:
            mn = value
            mnpos = time_pos
        if lookformax:
            if value < mx-DELTA:
                # a local maxima
                maxpeaks.append((mxpos, mx))
                mn = value
                mnpos = time_pos
                lookformax = False
            elif start:
                # a local minima at beginning
                minpeaks.append((mnpos, mn))
                mx = value
                mxpos = time_pos
                start = False
        else:
            if value > mn+DELTA:
                # a local minima
                minpeaks.append((mnpos, mn))
                mx = value
                mxpos = time_pos
                lookformax = True
    # check for extrema at end
    if value > mn+DELTA:
        maxpeaks.append((mxpos, mx))
    elif value < mx-DELTA:
        minpeaks.append((mnpos, mn))
    return minpeaks, maxpeaks


def c_excoef( commonfol , basefol ,  requiredStationsFile , stationsDB ):
    
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
            #print(station)
            station_names_for_comp.append(station)
        #elif station not in df_meas.columns:
            #print(station + ' does not have measured data')
        #elif station not in df_simul.columns:
            #print(station + ' does not have simulated data')
        #elif station not in stations:
            #print(station + ' does not have enough data in StationsDatabase')


    #check for extracted coefficients (to avoid re-extracting coefficients)
    #read stations names in the measured coefficients (if available)
#    try:
#        simul_coef_stations = pd.read_csv(path.join(basefol , 'coef_simulated' , 'simul_amplitude_all_stations.dat')).columns
#        meas_coef_stations = pd.read_csv(path.join(commonfol , 'coef_measured' , 'meas_amplitude_all_stations.dat')).columns
#    except :
#        print('no previous coefficient are generated')
#        simul_coef_stations = []
#        meas_coef_stations = []
#    
#    stations_for_ext = []
#    for station in station_names_for_comp:
#        if (station not in simul_coef_stations) or (station not in meas_coef_stations):
#            stations_for_ext.append(station)
#        
        
#    df_meas_crop = df_meas[period_s : period_e ]
#    df_simul_crop = df_simul[period_s : '2015-01-10' ]
#    date_inter = df_simul_crop.index.intersection(df_meas_crop.index)
    
#    datenum_meas = list(map(datenumaz,df_meas_crop.index.tolist()))
#    datenum_simul = list(map(datenumaz,df_simul_crop.index.tolist()))
    
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
    
#    def dt2dn(dt):
#       ord = dt.toordinal()
#       mdn = dt + datetime.timedelta(days = 366)
#       frac = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
#       return mdn.toordinal() + frac
        
    
    datesforpeaks = dfmeassimulforcomp.index 
    #and changing the index to datenum
    #dfmeassimulforcomp.index = dfmeassimulforcomp.index.map(datenumaz)
    
    #datenum = dfmeassimulforcomp.index
    datenum = dfmeassimulforcomp.index.map(datenumaz)

    #required tides avreviations
    pTides = ['MM','MF','Q1','O1','K1','SO1','MU2','N2','NU2','M2','S2','2SM2','MO3','MN4','M4'
          ,'MS4','MK4','S4','M6','2MS6','S6','M8','M10','M12' ] 
    #pTides = ['MM','MF','Q1','O1','K1','SO1','MU2','N2','NU2','M2','S2','2SM2','MO3','MN4','M4'
    #          ,'MS4','MK4','S4','M6','2MS6','S6','M8','M10','M12' ]
   
    #making directories for coefficients
    if not os.path.exists(os.path.join(basefol , 'coef_simulated')):
        os.makedirs(os.path.join(basefol , 'coef_simulated'))
    path_s = os.path.join(basefol , 'coef_simulated')
    
    if not os.path.exists(os.path.join(commonfol , 'coef_measured')):
        os.makedirs(os.path.join(commonfol , 'coef_measured'))
    path_m = os.path.join(commonfol , 'coef_measured')
    
    dfa = pd.DataFrame()
    dfg = pd.DataFrame()
    idx = pd.IndexSlice
    df_recons_h_meas = pd.DataFrame(columns = station_names_for_comp , index = datesforpeaks)
    df_recons_h_simul = pd.DataFrame(columns = station_names_for_comp , index = datesforpeaks)
    station_no_nan = []
    
    rmse_v_max_t = []
    rmse_v_min_t = [] 
    rmse_h_max_t = [] 
    rmse_h_min_t = [] 
    
#    np_recons_h_meas = np.empty((len(datenum) , len(station_names_for_comp)) )
#    np_recons_h_simul = np.empty((len(datenum) , len(station_names_for_comp)) )
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    print('Extracting coefficients and reconstructed water levels ...')
    #i=0
    n=0
    for station in station_names_for_comp:
        
        latitude = maindatabase.Latitude[station]
        
        n+=1
        print('station {} of {} ...'.format( n, len(station_names_for_comp)) )
        
        if dfmeassimulforcomp.loc[:,idx[station ,['meas']]].isnull().all().bool() == False:
            #MonteCarlo , ols
            #method = 'ols' , conf_int = 'MC'
            
            #tempcoefmeas = dfmeassimulforcomp.loc[:,idx[station ,['meas']]].apply(lambda x : (utide.solve(datenum , x , lat = latitude  , constit = pTides)))
            print('-------------------------------------')
            print(station+' ...coefficient calcs ...measured values ...')
            coef_meas = utide.solve(np.array(datenum) , dfmeassimulforcomp.loc[:,idx[station ,['meas']]].values[:,0] , lat = latitude , constit = pTides)
            #couldn't find how to save a coef
            #so i decided to merge the get_peaks function into this script
            #tempcoefsimul = dfmeassimulforcomp.loc[:,idx[station ,['simul']]].apply(lambda x : (utide.solve(datenum , x , lat = latitude ,  constit = pTides)))
            print('-------------------------------------')
            print(station+' ...coefficient calcs ...simulated values ...')
            coef_simul = utide.solve(np.array(datenum) , dfmeassimulforcomp.loc[:,idx[station ,['simul']]].values[:,0] , lat = latitude , constit = pTides)
            
            #reconstructing the coef for the peak comparison 
            print('-------------------------------------')
            print(station+' ...reconstructed water levels calcs ...measured values ...')
            recons_coef_meas = utide.reconstruct(np.array(datenum) , coef_meas)
            df_recons_h_meas[station] = recons_coef_meas['h']
            #np_recons_h_meas[: , i] = recons_coef_meas['h']
            print('-------------------------------------')
            print(station+' ...reconstructed water levels calcs ...measured values ...')
            recons_coef_simul = utide.reconstruct(np.array(datenum) , coef_simul)
            df_recons_h_simul[station] = recons_coef_simul['h']
            #np_recons_h_simul[: , i] = recons_coef_simul['h']
            
            #tempcoefmeas.to_csv(path.join(path_m , 'coef_'+station+'.dat'))
            #tempcoefsimul.to_csv(path.join(path_s , 'coef_'+station+'.dat'))
            
            #measindex = list(tempcoefmeas[station , 'meas']['name'])
            measindex = coef_meas['name'].tolist()
            #simulindex = list(tempcoefsimul[station , 'simul']['name'])
            simulindex = coef_simul['name'].tolist()
            
            #tempdfameas = tempcoefmeas.loc[idx['A'] , :].apply(pd.Series).T
            tempdfameas = pd.Series(coef_meas['A']).to_frame()
            tempdfameas.index = measindex
            tempdfameas.columns = [station] 
            tempdfameas.columns = pd.MultiIndex.from_product([tempdfameas.columns, ['meas']])
            #tempdfasimul = tempcoefsimul.loc[idx['A'] , :].apply(pd.Series).T
            tempdfasimul = pd.Series(coef_simul['A']).to_frame()
            tempdfasimul.index = simulindex 
            tempdfasimul.columns = [station] 
            tempdfasimul.columns = pd.MultiIndex.from_product([tempdfasimul.columns, ['simul']])
            
            dfa = pd.concat([dfa , tempdfameas] , axis = 1, sort = True)
            dfa = pd.concat([dfa , tempdfasimul] , axis = 1, sort = True)
            
            #tempdfgmeas = tempcoefmeas.loc[idx['g'] , :].apply(pd.Series).T
            tempdfgmeas = pd.Series(coef_meas['g']).to_frame()
            tempdfgmeas.index = measindex
            tempdfgmeas.columns = [station] 
            tempdfgmeas.columns = pd.MultiIndex.from_product([tempdfgmeas.columns, ['meas']])
            #tempdfgsimul = tempcoefsimul.loc[idx['g'] , :].apply(pd.Series).T
            tempdfgsimul = pd.Series(coef_simul['g']).to_frame()
            tempdfgsimul.index = simulindex    
            tempdfgsimul.columns = [station] 
            tempdfgsimul.columns = pd.MultiIndex.from_product([tempdfgsimul.columns, ['simul']])
            
            dfg = pd.concat([dfg , tempdfgmeas] , axis = 1, sort = True)
            dfg = pd.concat([dfg , tempdfgsimul] , axis = 1, sort = True)
            
            #i+=1
            station_no_nan.append(station)
            
            print('-------------------------------------')
            print(station+' ...finding peaks calcs ...')

            #finding peaks
            #in the following part, 2 represents simulated values and 3 represents the measured ones
            #simul before reconstruction 
            #DELTA = 0.3 (7 hours)
            minpeaks2, maxpeaks2 = findpeaks(dfmeassimulforcomp.loc[:,idx[station ,['simul']]].iloc[:, 0] , DELTA=0.3)
            
            fig, ax = plt.subplots()
            ax.set_ylabel('water level')
            ax.set_xlabel('Time')
            ax.set_title('Peaks in TimeSeries, simul before reconstruction')
            dfmeassimulforcomp.loc[:,idx[station ,['simul']]].iloc[:, 0] .plot()
            ax.scatter(*zip(*minpeaks2), color='red', label='min')
            ax.scatter(*zip(*maxpeaks2), color='green', label='max')
            ax.legend()
            ax.grid(True)
            plt.show()
            
            #meas before reconstruction
            minpeaks3, maxpeaks3 = findpeaks(dfmeassimulforcomp.loc[:,idx[station ,['meas']]].iloc[:, 0] , DELTA=0.3)
            
            fig, ax = plt.subplots()
            ax.set_ylabel('water level')
            ax.set_xlabel('Time')
            ax.set_title('Peaks in TimeSeries, meas before reconstruction')
            dfmeassimulforcomp.loc[:,idx[station ,['meas']]].iloc[:, 0].plot()
            ax.scatter(*zip(*minpeaks3), color='red', label='min')
            ax.scatter(*zip(*maxpeaks3), color='green', label='max')
            ax.legend()
            ax.grid(True)
            plt.show()
            
            #meas after reconstruction
            minpeaks3r, maxpeaks3r = findpeaks(df_recons_h_meas[station] , DELTA=0.3)

            fig, ax = plt.subplots()
            ax.set_ylabel('water level')
            ax.set_xlabel('Time')
            ax.set_title('Peaks in TimeSeries, meas after rcs')
            df_recons_h_meas[station].plot()
            ax.scatter(*zip(*minpeaks3r), color='red', label='min')
            ax.scatter(*zip(*maxpeaks3r), color='green', label='max')
            ax.legend()
            ax.grid(True)
            plt.show()
            
            #simul after reconstruction
            minpeaks2r, maxpeaks2r = findpeaks(df_recons_h_simul[station] , DELTA=0.3)

            fig, ax = plt.subplots()
            ax.set_ylabel('water level')
            ax.set_xlabel('Time')
            ax.set_title('Peaks in TimeSeries, simul after rcs')
            df_recons_h_simul[station].plot()
            ax.scatter(*zip(*minpeaks2r), color='red', label='min')
            ax.scatter(*zip(*maxpeaks2r), color='green', label='max')
            ax.legend()
            ax.grid(True)
            plt.show()
            
            #extracting locations of max and min peaks, before and after reconstruction
            maxlcs2 = []
            for i in range(len(maxpeaks2)):
                maxlcs2.append(maxpeaks2[i][0])
            
            minlcs2 = []
            for i in range(len(minpeaks2)):
                minlcs2.append(minpeaks2[i][0])
            
            maxlcs3 = []
            for i in range(len(maxpeaks3)):
                maxlcs3.append(maxpeaks3[i][0])
            
            minlcs3 = []
            for i in range(len(minpeaks3)):
                minlcs3.append(minpeaks3[i][0])
            
            
            
            maxlcs2r = []
            for i in range(len(maxpeaks2r)):
                maxlcs2r.append(maxpeaks2r[i][0])
            
            minlcs2r = []
            for i in range(len(minpeaks2r)):
                minlcs2r.append(minpeaks2r[i][0])
            
            maxlcs3r = []
            for i in range(len(maxpeaks3r)):
                maxlcs3r.append(maxpeaks3r[i][0])
            
            minlcs3r = []
            for i in range(len(minpeaks3r)):
                minlcs3r.append(minpeaks3r[i][0])
              
            #getting indices based in the reconstructed values    
            Dmax2 = cdist(np.array(list(map(datenumaz,maxlcs2r))).reshape(-1,1) , np.array(list(map(datenumaz,maxlcs2))).reshape(-1,1))
            Dmin2 = cdist(np.array(list(map(datenumaz,minlcs2r))).reshape(-1,1) , np.array(list(map(datenumaz,minlcs2))).reshape(-1,1))
            
            Dmax3 = cdist(np.array(list(map(datenumaz,maxlcs3r))).reshape(-1,1) , np.array(list(map(datenumaz,maxlcs3))).reshape(-1,1))
            Dmin3 = cdist(np.array(list(map(datenumaz,minlcs3r))).reshape(-1,1) , np.array(list(map(datenumaz,minlcs3))).reshape(-1,1))
            
            (indxmax2r , indxmax2) = np.where(Dmax2 == np.min(Dmax2 , axis=0) )
            (indxmax3r , indxmax3) = np.where(Dmax3 == np.min(Dmax3 , axis=0) )
            
            (indxmin2r , indxmin2) = np.where(Dmin2 == np.min(Dmin2 , axis=0) )
            (indxmin3r , indxmin3) = np.where(Dmin3 == np.min(Dmin3 , axis=0) )
            
            
            #dataframes for high and low water levels (max and min): index - location of peak - value of peak
            df_max2 = pd.DataFrame(data = maxpeaks2 , columns = ['maxsimullcs' , 'maxsimulpeaks'] ,  index = indxmax2r)

            df_min2 = pd.DataFrame(data = minpeaks2 , columns = ['minsimullcs' , 'minsimulpeaks'] ,  index = indxmin2r)
            
            df_max3 = pd.DataFrame(data = maxpeaks3 , columns = ['maxmeaslcs' , 'maxmeaspeaks'] ,  index = indxmax3r)
            
            df_min3 = pd.DataFrame(data = minpeaks3 , columns = ['minmeaslcs' , 'minmeaspeaks'] ,  index = indxmin3r)
            
            #joined dataframes
            df_max = df_max2.join(df_max3 , how = 'inner')
            df_min = df_min2.join(df_min3 , how = 'inner')
            
            
            #rmse calc
            rmse_v_max = 0.5**mean_squared_error(df_max['maxmeaspeaks'] , df_max['maxsimulpeaks'])
            rmse_v_min = 0.5**mean_squared_error(df_min['minmeaspeaks'] , df_min['minsimulpeaks'])
            
            rmse_h_max = 0.5**mean_squared_error(list(map(datenumaz,df_max['maxmeaslcs'].tolist())) , list(map(datenumaz,df_max['maxsimullcs'].tolist())) )
            rmse_h_min = 0.5**mean_squared_error(list(map(datenumaz,df_min['minmeaslcs'].tolist())) , list(map(datenumaz,df_min['minsimullcs'].tolist())) )

            #rmse for stations
            rmse_v_max_t.append(rmse_v_max)
            rmse_v_min_t.append(rmse_v_min)
            rmse_h_max_t.append(rmse_h_max)
            rmse_h_min_t.append(rmse_h_min)
            
            
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

    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    print('Partial tides comparison plots ...')
    
    n=0
    #looping through tides
    for tide in pTides:
        
        n+=1
        print('tide {} of {} ...'.format( n, len(pTides)) )
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
        
        print(tide +' ...amplitude comaprison extracted')
        print('-------------------------------------')
    
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
        
        print(tide +' ...phase shift comaprison extracted')
        print('-------------------------------------')


    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    #plotting rmse of high and low tides
    #high tides
    #subplot1 vertical, subplot2 horizontal; amongst stations
        
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False,figsize=(15,11))
    
    ax1.plot(station_no_nan ,  rmse_v_max_t  , 'x')
    ax1.set_xticks(list(range(len(station_no_nan))))
    ax1.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
    ax1.set_ylabel('Vertical RMSE')
    ax1.legend(['Vertical RMSE'])
    ax1.set_title('RMSE - High tides peaks values')

    plt.subplots_adjust(hspace=0.5)
    
    ax2.plot(station_no_nan ,  rmse_h_max_t  , 'x')
    ax2.set_ylabel('Horizontal RMSE')
    ax2.set_xlabel('Stations')
    ax2.legend(['Horizontal RMSE'])
    ax2.set_title('RMSE - High tides peaks locations')
    ax2.set_xticks(list(range(len(station_no_nan))))
    ax2.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
        
    savingname = path.join(path_1 , 'high_tides_rmse' + '.png')
    fig.savefig(savingname)
    plt.close()
    print('RMSE for high tides ... extracted ...')

    #low tides
    #subplot1 vertical, subplot2 horizontal; amongst stations
        
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False,figsize=(15,11))
    
    ax1.plot(station_no_nan ,  rmse_v_min_t  , 'x')
    ax1.set_xticks(list(range(len(station_no_nan))))
    ax1.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
    ax1.set_ylabel('Vertical RMSE')
    ax1.legend(['Vertical RMSE'])
    ax1.set_title('RMSE - Low tides peaks values')

    plt.subplots_adjust(hspace=0.5)
    
    ax2.plot(station_no_nan ,  rmse_h_min_t  , 'x')
    ax2.set_ylabel('Horizontal RMSE')
    ax2.set_xlabel('Stations')
    ax2.legend(['Horizontal RMSE'])
    ax2.set_title('RMSE - Low tides peaks locations')
    ax2.set_xticks(list(range(len(station_no_nan))))
    ax2.set_xticklabels(labels = station_no_nan,rotation=45 , horizontalalignment='right')
        
    savingname = path.join(path_1 , 'low_tides_rmse' + '.png')
    fig.savefig(savingname)
    plt.close()
    print('RMSE for low tides ... extracted ...')
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
