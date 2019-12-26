#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 04:49:47 2019

@author: amrozeidan
"""

import pandas as pd

def teldict(tel_module):
    """
    return telemac variable dictionary based on telemac module
    
    tel_module: can be either '2D' or '3D' or 'TOMAWAC'
    """

    
    teldict2d = {    'U':['VELOCITY U','velocity_u'] ,
                   'V' :['VELOCITY V','velocity_v']  ,
                  'C'  :['CELERITY' , 'celerity']   ,
                    'H':['WATER DEPTH','water_depth']  ,
                   'S' :['FREE SURFACE','free_surface'] ,
                    'B':['BOTTOM','bottom'] ,
                   'F' :['FROUDE NUMBER' , 'froude_number']  ,
                    'Q':['SCALAR FLOWRATE' , 'scalar_flowrate'] ,  
                    'SLNT':['SALINITY','salinity']  ,
                   'K' :['TURBULENT ENERG.' , 'turbulent_energie']   ,
                   'E' :['DISSIPATION' , 'dissipation']   ,
                    'D':['LONG. DISPERSION' , 'longitudinal_dispersion']   ,
                   'VS' :['VISCOSITY' , 'viscosity']   ,
                    'I':['FLOWRATE ALONG X' , 'flow_rate_x']  ,
                   'J'  :['FLOWRATE ALONG Y' , 'flow_rate_y'] ,
                    'M':['SCALAR VELOCITY' , 'scalar_velocity']  ,
                   'X' :['WIND ALONG X','wind_x']  ,
                    'Y':['WIND ALONG Y','wind_y'] ,
                   'P' :['AIR PRESSURE','air_pressure']  ,
                   'W' :['BOTTOM FRICTION' , 'bottom_friction']  ,
                    'A':['DRIFT ALONG X' , 'drift_x'] ,
                   'G' :['DRIFT ALONG Y' , 'drift_y'] ,
                   'L' :['COURANT NUMBER' , 'courant_number']  ,
                   'MAXZ' :['HIGH WATER MARK' , 'high_water_mark']  ,
                   'TMXZ' :['HIGH WATER TIME' , 'high_water_time']  ,
                    'MAXV':['HIGHEST VELOCITY' , 'highest_velocity']  ,
                   'TMXV' :['TIME OF HIGH VEL' , 'time_of_highest_velocity'] ,
                    'US' :['FRICTION VEL.' , 'friction_velocity'] ,
                   'TAU_S'  :['TAU_S' , 'tau_s']  ,
                    '1/R' :['1/R' , 'one_over_R']  ,
                   'WDIST' :['WALLDIST' , 'wall_dist'] }
    
    teldict3d = {   'Z' :['ELEVATION Z' , 'elevation_z'] , 
                    'U' :['VELOCITY U','velocity_u'] , 
                     'V':['VELOCITY V','velocity_v' ], 
                     'W':['VELOCITY W' , 'velocity_w'] , 
                     'NUX':['NUX FOR VELOCITY' , 'nux_velocity'] , 
                     'NUY' :['NUY FOR VELOCITY' , 'nuy_velocity'] ,  
                    'NUZ' :['NUZ FOR VELOCITY' , 'nuz_velocity'] , 
                     'K'  :['TURBULENT ENERGY' , 'turbulent_energy'] , 
                     'EPS'  :['DISSIPATION' , 'dissipation'] , 
                     'RI' :['RICHARDSON NUMB' , 'richardson_number'] , 
                    'RHO' :['RELATIVE DENSITY' , 'relative_density'] , 
                     'DP':['DYNAMIC PRESSURE' , 'dynamic_pressure'] ,  
                    'PH' :['HYDROSTATIC PRES' , 'hydrostatic_pressure'] ,  
                    'UCONV'  :['U ADVECTION' , 'advection_u'] , 
                     'VCONV' :['V ADVECTION' , 'advection_v'] , 
                      'WCONV'  :['W ADVECTION' , 'advection_w'] ,
                     '?' :['OLD VOLUMES' , 'old_volumes'] ,  
                     'DM1'  :['DM1' , 'dm1'] , 
                     'DHHN':['DHHN' , 'dhhn'] , 
                     'UCONVC' :['UCONVC' , 'convc_u'] , 
                     'VCONVC' :['VCONVC' , 'convc_v'] , 
                    'UD' :['UD', 'ud'] ,  
                     'VD':['VD' , 'vd'] ,  
                     'WD':['WD' , 'wd'] }
    teldicttomawac = {   'M0' :['VARIANCE M0' , 'variance_m0'] , 
                    'HM0' :['WAVE HEIGHT HM0','wave_height'] , 
                     'DMOY':['MEAN DIRECTION','wave_mean_direction' ], 
                     'SPD':['WAVE SPREAD' , 'wave_spread'] , 
                     'ZF':['BOTTOM' , 'bottom'] , 
                     'WD' :['WATER DEPTH' , 'water_depth'] ,  
                    'UX' :['VELOCITY U' , 'velocity_u' ] , 
                     'UY'  :['VELOCITY V' , 'velocity_v'] , 
                     'FX'  :['FORCE FX' , 'force_fx'] , 
                     'FY' :['FORCE FY' , 'force_fy'] , 
                    'SXX' :['STRESS SXX' , 'stress_sxx'] , 
                     'SXY':['STRESS SXY' , 'stress_sxy'] ,  
                    'SYY' :['STRESS SYY' , 'stress_syy'] ,  
                    'UWB'  :['BOTTOM VELOCITY' , 'bottom_velocity'] , 
                     'PRI' :['PRIVATE 1' , 'private_one'] , 
                      'FMOY'  :['MEAN FREQ FMOY' , 'mean_frequency_y'] ,
                     'FM01' :['MEAN FREQ FM01' , 'mean_frequency_one'] ,  
                     'FM02'  :['MEAN FREQ FM02' , 'mean_frequency_two'] , 
                     'FPD':['PEAK FREQ FPD' , 'peak_frequency'] , 
                     'FPR5' :['PEAK FREQ FPR5' , 'peak_frequency_five'] , 
                     'FPR8' :['PEAK FREQ FPR8' , 'peak_frequency_eight'] , 
                    'US' :['USTAR', 'usar'] ,  
                     'CD':['CD' , 'cd' ] ,  
                     'Z0':['Z0 ' , 'z_zero'],
                     'WS'  :['WAVE STRESS' , 'wave_stress'] , 
                     'TMOY' :['MEAN PERIOD TMOY' , 'wave_mean_period'] , 
                      'TM01'  :['MEAN PERIOD TM01' , 'wave_mean_period_one'] ,
                     'TM02' :['MEAN PERIOD TM0' , 'wave_mean_period_two'] ,  
                     'TPD'  :['PEAK PERIOD TPD' , 'wave_peak_period'] , 
                     'TPR5':['PEAK PERIOD TPR5' , 'wave_peak_period_five'] , 
                     'TPR8' :['PEAK PERIOD TPR8' , 'wave_peak_period_eight'] , 
                     'POW' :['WAVE POWER' , 'wave_power'] , 
                    'BETA' :['BETA' , 'beta']}

    if tel_module == '2D':   
        return teldict2d 
    elif tel_module == '3D': 
        return teldict3d 
    elif tel_module == 'TOMAWAC':
        return teldicttomawac
















