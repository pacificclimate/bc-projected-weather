import io
import numpy
import pandas
import glob
from datetime import datetime
from netCDF4 import Dataset

from bcweather import get_epw_header, get_climate_data, get_ensemble_averages, format_netcdf_series
from bcweather import get_epw_summary_values, gen_prism_offset_weather_file, morph_dry_bulb_temperature
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

def test_get_ensemble_averages():
    print('Test Ensemble Alpha Delta')
    cdfvariable = 'dewpoint'
    rcp = 'rcp85'
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/"
    ##"/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/"
    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
    glen = len(gcms)
    past_files = []
    proj_files = []
    for gcm in gcms:
        past_file = glob.glob(gcm_dir + gcm + '/' + cdfvariable + '_day_*' + rcp + '*19500101-21001231.nc') ##'*1951-2000.nc')
        proj_file = glob.glob(gcm_dir + gcm + '/' + cdfvariable + '_day_*' + rcp + '*19500101-21001231.nc') ##'*2001-2100.nc')
        past_files.append(past_file[0])
        proj_files.append(proj_file[0])
    factor = 'monthly'
    tasmax_present = get_ensemble_averages(cdfvariable=cdfvariable,
                                           lon=-122.36,lat=49.03,
                                           gcm_files=past_files,
                                           time_range=[1971,2000],
                                           factor=factor)
    past_mean = tasmax_present['std']
    print(past_mean.shape)
    tasmax_future = get_ensemble_averages(cdfvariable=cdfvariable,
                                          lon=-122.36,lat=49.03,
                                          gcm_files=proj_files,
                                          time_range=[2041,2070],
                                          factor=factor)
    assert(past_mean.shape == (12,len(gcms)))


def test_epw_to_data_frame(epwfile):
    print('EPW to Data Frame')
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns


def test_get_epw_summary_values(epwfile):
    print('Get EPW Summary Values')
    df = epw_to_data_frame(epwfile)
    epw_tas = numpy.array(df['dry_bulb_temperature'])
    epw_dates = df['datetime']
    epw_months = pandas.DatetimeIndex(epw_dates).month
    y = get_epw_summary_values(df['dry_bulb_temperature'], df['datetime'],
                               '%Y-%m-%d','max')
    z = get_epw_summary_values(y['data'],y['datetime'],'%Y %m %d','mean')
    assert z.shape == (365,2)

def test_gen_prism_offset_weather_file():
    gen_prism_offset_weather_file(49.2,-123.2)
    assert 1 == 1


def test_format_netcdf_series():
    print(' ')
    print('Test Format Netcdf Series')
    time_range = [1971,1972]
    calendar = 'gregorian'
    # data = numpy.arange(1,731,1)
    data = numpy.arange(1,732,1)
    # data = numpy.arange(1,721,1)
    test = format_netcdf_series(time_range,calendar,data)
    print(type(test['Time']))
    print(type(test['Data']))
    assert type(test['Time']) == list
