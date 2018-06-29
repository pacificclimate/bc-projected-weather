import io

import pytest
import numpy
import pandas

from bcweather import get_epw_header, get_climate_data
from bcweather import get_monthly_values, gen_prism_offset_weather_file, gen_future_weather_file, stretch_gcm_variable
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


##def test_get_climate_data(ncfile):
##    print(' ')
##    print('Test get climate data')
##    data = get_climate_data(ncfile, 50.8, -118.38, 'tasmax', [1970, 1971])
##    print(data)
##    print(numpy.array(data).shape)
##    print(len(data))
##    assert data.any()


def test_epw_to_data_frame(epwfile):
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns


##def test_get_monthly_values(epwfile):
##    df = epw_to_data_frame(epwfile)
##    x = get_monthly_values(df['dry_bulb_temperature'], df['datetime'])
##    print(x)
##    assert x.shape == (12,3)


##def test_gen_prism_offset_weather_file():
##    gen_prism_offset_weather_file(49.2,-123.2)
##    assert 1 == 1

def test_gen_future_weather_file():
    print('Test future weather file production')
    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
    x = gen_future_weather_file(epw_filename="/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw",
                                epw_output_filename="/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/future_test.epw",
                                present_range=[1971,2000],
                                future_range=[2041,2070],
                                gcms=gcms)
    assert type(x) == str

##def test_stretch_gcm_variable(epwfile):
##    print('Test GCM stretch with GCM ensemble')
##    df = epw_to_data_frame(epwfile)    
##    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
##    x = stretch_gcm_variable(epw_data=df,
##                             lon=-122.36,lat=49.03,
##                             epw_variable="relative_humidity",
##                             netcdf_variable="psl",
##                             gcms=gcms,
##                             present_range=[1971,2000],
##                             future_range=[2041,2070])
##    assert type(x) == pandas.DataFrame    
