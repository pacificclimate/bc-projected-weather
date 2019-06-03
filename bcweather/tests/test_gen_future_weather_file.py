import io
import numpy as np
import pandas
import glob
import pdb

from netCDF4 import Dataset
from bcweather import get_epw_header, get_ensemble_averages, get_climate_data
from bcweather import get_epw_summary_values
from bcweather import offset_current_weather_file
from bcweather.epw import epw_to_data_frame


def test_get_epw_header():
    my_string = """Line 1
Line 2
More stuff
Other stuff
Still more worthless data
What else could *possibly* in this file?!
Line 7
Line 8
Line 9
"""
    f = io.StringIO(my_string)
    pos = 15
    f.seek(pos)
    rv = get_epw_header(f)
    assert rv.startswith("Line 1")
    assert rv.endswith("Line 8\n")
    assert len(rv.splitlines()) == 8
    assert f.tell() == pos

def test_get_climate_data():
    print('Test Read Alpha and Delta Tas')

    rcp = 'rcp85'
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0",
           "GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
           "MIROC5", "MRI-CGCM3"]
    alpha_files = []
    delta_files = []
    for gcm in gcms:
        print(gcm)
        alpha_file = glob.glob(gcm_dir + 'alpha_tasmax_tasmin_'
                              + gcm + '_1971-2000_2041-2070.nc')
        delta_file = glob.glob(gcm_dir + 'delta_tas_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
        delta_files.append(delta_file[0])

    print(alpha_files)
    print(delta_files)
    fac = '%m'
    tlen = 12
    alpha_tas = np.zeros((tlen, len(gcms)))
    delta_tas = np.zeros((tlen, len(gcms)))

    for i, gcm_file in enumerate(alpha_files):
        with Dataset(gcm_file) as f:
            alphas = get_climate_data(
                f, lat=49.249541, lon=-123.015469, cdfvariable='alpha_tas', factor=fac)

        print(alphas['mean'])
        print(alphas['mean'].shape)
        print(alphas['mean'].flatten())
        print(alphas['mean'].flatten().shape)
        alpha_tas[:, i] = alphas['mean'].flatten()
    ##for i, gcm_file in enumerate(delta_files):
    ##    with Dataset(gcm_file) as f:
    ##        deltas = get_climate_data(
    ##            f, lat=49.03, lon=-122.36, cdfvariable='tas', factor=fac)
    ##    delta_tas[:, i] = deltas['mean'].flatten()

    print(alpha_tas.shape)
    ##print(delta_tas.shape)
    assert(alpha_tas.shape == (12, len(gcms)))
    ##assert(delta_tas.shape == (12, len(gcms)))

##"""
def test_get_ensemble_averages():
    print('Test Ensemble Alpha Delta')
    cdfvariable = 'alpha_tas'
    rcp = 'rcp85'
    factor = 'monthly'
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0",
            "GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
            "MIROC5", "MRI-CGCM3"]
    alpha_files = []
    for gcm in gcms:
        alpha_file = glob.glob(gcm_dir + 'alpha_tasmax_tasmin_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
        print(alpha_file)
    alpha_tas = get_ensemble_averages(cdfvariable=cdfvariable,
                                      lon=-123.015469, lat=49.249541,
                                      gcm_files=alpha_files,
                                      time_range=[1971, 2000],
                                      factor=factor,rlen=21)    
    print(alpha_tas['mean'].shape)
    print(alpha_tas['mean'])
    assert(len(alpha_tas['mean']) == 12)
##"""

def test_epw_to_data_frame(epwfile):
    print('EPW to Data Frame')
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns


def test_get_epw_summary_values(epwfile):
    print('Get EPW Summary Values')
    df = epw_to_data_frame(epwfile)
    y = get_epw_summary_values(df['dry_bulb_temperature'], df['datetime'],
                               '%Y-%m-%d', 'max')
    z = get_epw_summary_values(y['data'], y['datetime'], '%Y %m %d', 'mean')
    assert z.shape == (365, 2)


def test_offset_current_weather_file():
    offset_current_weather_file(-123.2,49.2,"UVic",
                                "/storage/data/projects/rci/weather_files/wx_files/",
                                "/storage/data/projects/rci/weather_files/wx_files/morphed_files/")
    assert 1 == 1


