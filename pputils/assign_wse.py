#
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 assign_wse.py                         # 
#                                                                       #
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng.
# 
# Date: Dec 8, 2017
#
# Purpose: Script takes in a mesh in *.slf format and a PPUTILS polygon
# file (containing areas of constant water surface elevations) to produce 
# a *.slf file to be used as a warm start in a Telemac-2D simlation. 
# This script mirrors assign_h.py, except that it works when assigning
# water surface elevations to a mesh (the artithmetic is different)
#
# Uses: Python 2 or 3, Numpy
#
# Example:
#
# python assign_bc.py -m mesh.slf -p poly.csv -o mesh_warm_start.slf
# where:
#
# -m input mesh file in *.slf format
# -p input boundary file (where each polygon has an attribute value to 
#    be assigned to the mesh)
# -o output file in *.slf format that can be used as a warm start 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os,sys                              # system parameters
import numpy             as np             # numpy
from ppmodules.selafin_io_pp import *      # to get SELAFIN I/O 
from ppmodules.utilities import *          # to get the utilities (point_in_poly)
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
#
# I/O
if len(sys.argv) != 7:
  print('Wrong number of Arguments, stopping now...')
  print('Usage:')
  print('python assign_bc.py -m mesh.slf -p poly.csv -o mesh_warm_start.slf')
  sys.exit()

mesh_file = sys.argv[2]
poly_file = sys.argv[4]
output_file  = sys.argv[6]

# now read the input *.slf geometry file
slf = ppSELAFIN(mesh_file)
slf.readHeader()
slf.readTimes()

times = slf.getTimes()
variables = slf.getVarNames()
units = slf.getVarUnits()
float_type,float_size = slf.getPrecision()
NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = slf.getMesh()

# reads all vars at the last time step in the file
slf.readVariables(times[-1])
results = slf.getVarValues()

# gets the number of planes
NPLAN = slf.getNPLAN()

if (NPLAN > 1):
  print('3d SELAFIN files are not yet supported. Exiting!')
  sys.exit()

# make sure that the variable 'BOTTOM' is in the mesh_file
idx_bottom = -999

# find the index of the vector variables
# in case the *.slf file has both bottom, and bottom friction, take only bottom

for i in range(len(variables)):
  if (variables[i].find('BOTTOM') == 0) & (variables[i].find('BOTTOM FRICTION') !=0 ):
    idx_bottom = i

if (idx_bottom < 0):
  print('Variable BOTTOM not found in input file. Exiting!')
  sys.exit()



# this is the bottom array, as a 1d vector
bottom = results[idx_bottom,:]
  
# now we read the BC poly file using numpy
poly_data = np.loadtxt(poly_file, delimiter=',',skiprows=0,unpack=True)

# boundary data
shapeid_poly = poly_data[0,:]
x_poly = poly_data[1,:]
y_poly = poly_data[2,:]
attr_poly = poly_data[3,:]

# round boundary nodes to three decimals
x_poly = np.around(x_poly,decimals=3)
y_poly = np.around(y_poly,decimals=3)

# total number of nodes in the polygon file
nodes = len(x_poly)

# get the unique polygon ids
polygon_ids = np.unique(shapeid_poly)

# find out how many different polygons there are
n_polygons = len(polygon_ids)

# to get the attribute data for each polygon
attribute_data = np.zeros(n_polygons)
attr_count = -1

# go through the polygons, and assign attribute_data 
for i in range(nodes-1):
  if (shapeid_poly[i] - shapeid_poly[i+1] < 0):
    attr_count = attr_count + 1
    attribute_data[attr_count] = attr_poly[i]
    
# manually assign the attribute_data for the last polygon
attribute_data[n_polygons-1] = attr_poly[nodes-1]

# define the default attribute (i.e., water surface elevation)
wse = np.zeros(NPOIN)

# loop over all polygons
for i in range(n_polygons):
  # construct each polygon
  poly = []
  for j in range(nodes):
    if (shapeid_poly[j] == polygon_ids[i]):
      poly.append( (x_poly[j], y_poly[j]) )
  #print (poly)
  
  # to loop over all nodes in the *.slf file
  for k in range(NPOIN):
    poly_test = point_in_poly(x[k], y[k], poly)
    if (poly_test == 'IN'):
      wse[k] = attribute_data[i]
  
  # delete all elements in the poly list
  del poly[:]    
  
# now we are ready to write the new *.slf warm start (ws) file
slf_ws = ppSELAFIN(output_file)
slf_ws.setPrecision(float_type,float_size)
slf_ws.setTitle('warm start file created with pputils')
slf_ws.setVarNames(['BOTTOM','WATER DEPTH','FREE SURFACE','VELOCITY U', 'VELOCITY V'])
slf_ws.setVarUnits(['M','M','M','M/S','M/S'])
slf_ws.setIPARAM([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
slf_ws.setMesh(NELEM, NPOIN, NDP, IKLE, IPOBO, x, y)
slf_ws.writeHeader()

# re-create the depth array from the wse and bottom arrays
depth = np.subtract(wse, bottom)

# make sure there are not negative depths,
# and that wse is not smaller than bottom
for i in range(NPOIN):
  if (depth[i] < 0):
    depth[i] = 0.0
  if (wse[i] < bottom[i]):
    wse[i] = bottom[i]

# this is the master results to write for the warm start file
res_ws = np.zeros((5,NPOIN))
res_ws[0,:] = bottom
res_ws[1,:] = depth
res_ws[2,:] = wse
res_ws[3,:] = np.zeros(NPOIN)
res_ws[4,:] = np.zeros(NPOIN)

# it does not write the time of the original file, but rather
# writes zero instead as the time of the warm start file
slf_ws.writeVariables(0, res_ws)
