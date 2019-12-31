#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:43:48 2019

@author: amrozeidan
"""
#all imports
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

def a_extelmac(lib_func_fol , commonfol , basefol , stationsDB , slffile , reqvar , telmod):
    
    startt = time.time()
 
    sys.path.append(path.join(lib_func_fol , 'pputils' , 'ppmodules'))
    sys.path.append(path.join(lib_func_fol , 'gshtools')) 
    from selafin_io_pp import ppSELAFIN
    from TelDict import teldict
    
    
    #extracting data from all stations database
    #filename_infodata = '/Users/amrozeidan/Desktop/py_testing/com1/info_all_stations.dat'
    #filename_infodata = commonfol + os.sep + 'info_all_stations.dat'
    stations_data = np.loadtxt(stationsDB , delimiter=',',skiprows=1 , dtype = str)
    
    station_names = stations_data[:,0]
    stations = station_names.tolist()
    easting = stations_data[:,1].astype(float)
    northing = stations_data[:,2].astype(float)
    
    RW_HW = np.transpose(np.vstack((easting , northing)))
    
    
    # read *.slf file
    #slf = ppSELAFIN('/Users/amrozeidan/Documents/hiwi/scripts/allcomp/t2d___res1207_NSea_rev03a.4_y2006_conf00.slf')
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
    times = slf.getTimes()
    DT = (times[2] - times[1])/ 3600 #timestep in hours
    init_date = datetime.datetime(2015 , 1, 6 , 0 , 0)
    nsteps = len(slf.getTimes())
    date_array = []
    for i in range(nsteps):
        date_array_i = init_date + i*datetime.timedelta(hours = DT )
        date_array.append(date_array_i) 
    
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
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    print('Extracting plots ...')
    #looping through required stations
    n=0
    for station in station_names_for_comp:
        #excluding nan 
        dfstation = df.loc[idx[:] , idx[ station , ['meas' , 'simul']]] 
        dfstation = dfstation.dropna() #removing nan
        x =  dfstation[station].meas
        y =  dfstation[station].simul
        
        
        n+=1
        print('station {} of {} ...'.format( n, len(station_names_for_comp)) )
        
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
            
                        
            print(station+' ...water level comaprison extracted')
            print('-------------------------------------')
         
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
            
            print(station+' ...water level scatter plot extracted')
            print('-------------------------------------')
            
            #nrmse extracting and storing
            nrmse = rmse/(x.max()-x.min())        
            totnrmse.append(nrmse) 
            totrmse.append(rmse)
            totmae.append(mae)
            totme.append(me)
            
            station_no_nan.append(station)
            
        else:
            print(station + ' has no measured values (all are nan). It will not be included in plots and errors calcs')
            print('-------------------------------------')
            
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
        
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
    
    print('NRMSE of WL on the required locations ... extracted')
    print('-------------------------------------')
    
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
    
    print('RMSE,MAE and ME of WL on the required locations ... extracted')
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')


    
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
    
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    print('Extracting salinity comparison plots ...')
    
    n=0
    #comparison plots
    for station in station_names_for_comp:
        n+=1
        print('station {} of {} ...'.format( n, len(station_names_for_comp)) )
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
                
                #if (np.isnan(min_diff)==False) and (np.isnan(max_diff)==False):
                ax2.set_ylim(min_diff-1 , max_diff+1)
                ax2.set_title('Salinity difference,station: ' + station + ', depth= ' + i)  
                ax2.legend(['Difference'])
                ax2.set_xlabel('Date/Time')
                ax2.set_ylabel('Salinity difference')
                #plt.subplots_adjust(hspace=0.3)
                savingname = path.join(path_1 , 'salinity_comp_diffr_station_'+station + '_' + i + '.png')
                fig.savefig(savingname )
                plt.close() 
                
                print(station+' ...salinity comparison extracted ...')
                print('-------------------------------------')
    

    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    

def e_velocitycomp( commonfol , basefol , period_s , period_e  , requiredStationsFile ):
    
    station_names_req = np.loadtxt(requiredStationsFile , delimiter='\t', dtype = str).tolist()

#    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_simul = pd.read_csv(path.join(basefol, 'telemac_variables','variables_all_stations' ,'velocity_all_stations.dat' ) ,index_col =0 )
    df_simul.index = pd.to_datetime(df_simul.index)
#                           header =0 , parse_dates = ['Unnamed: 0'],date_parser = dateparse, index_col =0 , squeeze=True)
#    df_simul.set_index('Unnamed: 0', inplace = True)
    
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
#    idx_cp = df_simul.columns.str.split('_', expand=True)
#    df_simul.columns = idx_cp
    
    
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
    
#    df_for_nonan = df_meas.loc[: , pd.IndexSlice[station , i , j]].join(df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]], how = 'inner')
#    df_for_nonan = df_for_nonan.sort_index(axis = 1)
    df_for_nonan = pd.concat([df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]] , df_meas.loc[: , pd.IndexSlice[station , i , j]] ], axis = 1)
    
    date_plots = df_simul_meas.index.to_list()
#    [max_meas_simul , max_diff] = [ np.fmax( np.nanmax(df_meas.max()) , np.nanmax(df_simul.max()) ), np.nanmax(df_simul_meas.max())]
#    [min_meas_simul , min_diff] = [ np.fmin( np.nanmin(df_meas.min()) , np.nanmin(df_simul.min()) ), np.nanmin(df_simul_meas.min())]
    
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
    
    plot_dict = {'magn': 'Velocity Magnitude', 'dirc': 'Velocity Direction', 'velv': 'Velocity v' , 'velu': 'Velocity u'} 
    plot_dict_units = {'magn': '[m/s]', 'dirc': '[add units]', 'velv': '[m/s]' , 'velu': '[m/s]'} 
    
    plot_dict_min = {'magn': magn_min_meas_simul, 'dirc': dirc_min_meas_simul, 'velv': velv_min_meas_simul , 'velu': velu_min_meas_simul} 
    plot_dict_max = {'magn': magn_max_meas_simul, 'dirc': dirc_max_meas_simul, 'velv': velv_max_meas_simul , 'velu': velu_max_meas_simul} 
    
    plot_dict_min_diff = {'magn': magn_min_diff, 'dirc': dirc_min_diff, 'velv': velv_min_diff , 'velu': velu_min_diff} 
    plot_dict_max_diff = {'magn': magn_max_diff, 'dirc': dirc_max_diff, 'velv': velv_max_diff , 'velu': velu_max_diff} 
    
    
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    
    print('Extracting velocity components comparison plots ...')
    
    n=0
    #comparison plots
    for station in station_names_for_comp:
        
        n+=1
        print('station {} of {} ...'.format( n, len(station_names_for_comp)) )
        for i,j in df_meas[station].columns.to_list():
            
#            df_for_nonan = pd.concat([df_simul.loc[: , pd.IndexSlice[station , j , 'simul' ]] , df_meas.loc[: , pd.IndexSlice[station , i , j]] ], axis = 1)
#            df_for_nonan = df_for_nonan.dropna()
#            if not df_for_nonan.empty:    
#                x = df_for_nonan.iloc[1]
#                y = df_for_nonan.iloc[0]
#                mse = mean_squared_error(x,y)
#                rmse = mse**0.5 
#                me = np.mean(y-x)
#                mae = mean_absolute_error(x,y)
#            else:
#                mse = 0
#                rmse = 0
#                me = 0
#                mae = 0
            
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
                #if (np.isnan(plot_dict_min_diff[j])==False) and (np.isnan(plot_dict_max_diff[j])==False):
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
                
        print(station+' ...velocity components comparison extracted ...')
        print('-------------------------------------')
        
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')

def f_wavecomp( commonfol , basefol , period_s , period_e  ,  requiredStationsFile ):

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

def j_all(lib_func_fol , commonfol , basefol , stationsDB , requiredStationsFile , slffile , reqvar , telmod ,  period_s , period_e , k , *args ):
    '''
    This tool serves to:
        extract required variables from a .slf file (tomawac or 2d modules) for required stations (coordinates) 
        and then run a comparison between the extracted values (simulated values) and measured values for:
            water levels
            tides components (amplitude phase shift) after extracting them
            salinity
            velocity components (velocity u, velocity v, velocity magnitude and velocity direction)
            wave components (wave height swh , wave direction mwd, wave mean period mwp and wave peak period pwp)
        
        Parameters
        ----------
        lib_func_fol : folder directory
            Folder location which contains 'gshtools' and 'pputils' sub-folders.
        commonfol : folder directory
            Common folder location. This folder should contain 'measurements' sub-folder containing measured
            values of different variables amongst different stations (yyyy.cu.dat , yyyy.sa.dat , yyyy.wl.dat
             and yyyy.wv.dat)
        basefol: folder directory
            Base folder location. An empty folder where the results are stored.
        stationsDB : .dat file directory 
            File containing info about all stations like names, easting, northing, latitudes and numbering(this 
            numbering starts from France and ends up in England going counter clickwise around the North Sea)
        requiredStationsFile : .dat file directory
            File contaning the names of stations to be compared.
        slffile : .slf file directory 
            Selafin file directory.
        reqvar : array_like
            List of variables abreviations to be extracted from the .slf file (referr to TelDict.py for abrv)
            example : reqvar = ['U' , 'V' , 'S' , 'SLNT' , 'W' ,  'A' , 'G' , 'H']
        telmod : string
            Selafin module. '2D' for 2d module and 'TOMAWAC' for tomawac module.
        period_s : string
            Comparison starting period. example : period_s = '2015-01-06' 
        period_e : string
            Comparison ending period. example : period_e = '2015-01-06' 
        k : int
            Used for water level moving average calculations. k=147 default
        
        *args
        -----
        'a' : extract variables function
        'b' : water level comparison 
        'c' : extract partial tide coefficients and compare them
        'd' : salinity comparison
        'e' : velocity components comparison
        'f' : wave components coomparison
        
        returns
        -------
        in commfol : 'coef_measured' folder containing partial tides coeffitient for measured data
        in basefol :
            'telemac_variables' folder containing simulated values of extracted variables
            'wlcomp' folder containing .png files of water level comparison. for each station:
                -water level comparison and water level difference
                -water level density scatter plot
            for all stations:
                -NRMSE of water level
                -RMSE, MEA and ME of water level
            'ptcomp' folder containing files of partial tides comparison. for each tide:
                -amplitude comparison of required stations
                -phase shift comparison of required stations
            'coef_simulated' folder containing partial tides coeffitient for simulated data
            'salinitycomp' containing files of salinity comparison
            'velocitycomp' containing files of velocity comparison. for each station:
                -velocity u comparison
                -velocity v comparison
                -velocity direction comparison
                -velocity magnitude comparison
            'wavecomp' containing files of wave comparison. for each station:
                -wave height comparison
                -wave direction comparison
                -wave mean period comparison
                -wave peak period comparison
    
    '''
    
    if 'a' in args:
        a_extelmac(lib_func_fol , commonfol , basefol , stationsDB , slffile , reqvar , telmod)
    if 'b' in args:
        b_wlcomp( commonfol , basefol , period_s , period_e , k , requiredStationsFile )
    if 'c' in args:
        c_excoef( commonfol , basefol , period_s , period_e  , requiredStationsFile , stationsDB )
    if 'd' in args:
        d_salinitycomp( commonfol , basefol , period_s , period_e  ,  requiredStationsFile )
    if 'e' in args:
        e_velocitycomp( commonfol , basefol , period_s , period_e  , requiredStationsFile )
    if 'f' in args:
        f_wavecomp( commonfol , basefol , period_s , period_e  ,  requiredStationsFile )

if __name__=="__main__":
    j_all()      