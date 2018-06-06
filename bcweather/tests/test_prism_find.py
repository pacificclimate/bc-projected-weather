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

def prism_test():
    coords = (-123.2, 49.2)
    wx_closest = find_closest_wx_file(coords)
    print(wx_closest)
    wx_close_coords = get_epw_coordinates(wx_closest)
    print(wx_close_coords)
    nc = ncfile('pr')
    prism_cell = get_prism_indices(nc,coords)
    print('Closest PRISM cell to supplied coords')
    print(prism_cell)
    wx_cell = get_prism_indices(nc,wx_close_coords)
    print('PRISM coords of cell closest to WX File')
    print(wx_cell)
    return(prism_cell)
