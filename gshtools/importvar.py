#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:25:20 2019

@author: amrozeidan
"""

import pandas as pd

def importMeasWl (measwlfilename , locname):
    '''To import measured water level values from BAW dat file 
    input: measwlfilename; dat file directory '''
    
    lineno = 0
    with open(measwlfilename, 'r') as input:
       for line in input:
           lineno +=1
           if '# ------------------------------------------' in line:
               break
           
    measwldf = pd.read_csv(measwlfilename , header = lineno , 
                             sep= '\s+', engine='python' , squeeze = True,
                             names = ['date','time',locname,'cal'])
                             
                             
    measwldf['datetime'] = measwldf['date'].map(str) +' '+ measwldf['time'].str.replace(';','')
    measwldf = measwldf[[locname ,"datetime"]]
    measwldf['datetime'] = pd.to_datetime(measwldf['datetime'])
    measwldf.set_index('datetime' , inplace = True) 
    
    return measwldf

