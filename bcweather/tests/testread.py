from datetime import datetime
import glob
import numpy as np
import netcdf
from netCDF4 import Dataset
import netCDF4 as cdf
import pandas as pd

def get_epw_coordinates(filename):    
    nc = pd.read_csv(filename,sep=',',header=None,nrows=1)
    rv = (float(nc[7].values), float(nc[6].values)) ##FIXME Replace with pattern matching    
    return(rv)   

def list_of_epw_coordinates(files):
    coords = np.zeros((len(files),2))
    for i, file in enumerate(files):
        b = get_epw_coordinates(file)
        coords[i,:] = b
    return(coords)
    
def find_closest_wx_file(coords):
    read_dir = "/home/ssobie/Desktop/weather_files/wx_files/"
    files = glob.glob(read_dir+'*.epw')
    coord_data = list_of_epw_coordinates(files)
    ##print(coord_data)
    wx_index = np.sum(np.square(coord_data-coords),axis=1).argmin()    
    wx_selected = files[wx_index]
    return(wx_selected)

def ncfile(varname):
    fname = '/home/ssobie/Desktop/weather_files/PRISM/'+ varname + '_lm_subset.nc'
    dst = cdf.Dataset(fname, 'r')
    return dst

def get_prism_indices(nc, coords):
    lon_data = nc.variables["lon"][:]
    lon_index = np.absolute(lon_data - coords[0]).argmin()
    lat_data = nc.variables["lat"][:]
    lat_index = np.absolute(lat_data - coords[1]).argmin()
    rv = [lon_index,lat_index]
    return(rv)            

def prism_read(nc, cells, varname):
    data = nc.variables[varname][:, cells[0], cells[1]]
    return(data[0:12,])

def wx_read(filename):
    wx = pd.read_csv(filename,sep=',',header=None,skiprows=8)    
    return(wx)

def prism_tas(nc,cells):
    ncx = ncfile('tmax')
    tmax = prism_read(ncx,cells,'tmax')
    ncn = ncfile('tmin')
    tmin = prism_read(ncn,cells,'tmin')
    tas = np.divide(tmax + tmin,2.0)
    return(tas)

def prism_closest():
    coords = (-123.2, 49.2)
    varname='tmax'
    wx_closest = find_closest_wx_file(coords)
    print(wx_closest)

    wx_close_coords = get_epw_coordinates(wx_closest)
    print(wx_close_coords)
    nc = ncfile(varname)

    print('Closest PRISM cell to supplied coords')    
    prism_cell = get_prism_indices(nc,coords)
    print(prism_cell)
    prism_loc_tas = prism_tas(nc,prism_cell)
    print('PRISM coords of cell closest to WX File')
    wx_cell = get_prism_indices(nc,wx_close_coords)
    print(wx_cell)

    prism_wx_tas = prism_tas(nc,wx_cell)
    prism_diff = prism_loc_tas - prism_wx_tas
    
    print('Local')
    print(prism_loc_tas)
    print('WX')
    print(prism_wx_tas)
    print('Diff')
    print(prism_diff)

    wx_data = wx_read(wx_closest)
    wx_tas = wx_data[6]
    wx_dates = wx_data.iloc[:,0:3]
    return(wx_data)

##Data from epw file
##Dry bulb temp [6]
##


