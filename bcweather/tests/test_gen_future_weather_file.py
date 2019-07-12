import io
import numpy as np
import pandas
import glob
import matplotlib.pyplot as plt 
import os as os

from netCDF4 import Dataset
from bcweather import get_epw_header, get_ensemble_averages, get_climate_data
from bcweather import get_epw_summary_values, generate_dry_bulb_temperature
from bcweather import generate_dewpoint_temperature
from bcweather import generate_horizontal_radiation
from bcweather import generate_stretched_series
from bcweather import morph_atmospheric_station_pressure
from bcweather import morph_direct_normal_radiation
from bcweather import morph_relative_humidity
from bcweather import morph_wind_speed
from bcweather import morph_total_sky_cover
from bcweather import morph_opaque_sky_cover
from bcweather import offset_current_weather_file
from bcweather import gen_future_weather_file
from bcweather.epw import epw_to_data_frame


def test_epw_to_data_frame(epwfile):
    print('EPW to Data Frame')
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns


def test_get_epw_summary_values(epwfile):
    print('Get EPW Summary Values')
    df = epw_to_data_frame(epwfile)
    input_dict = {'datetime': df['datetime'],
                  'data': df['dry_bulb_temperature']}
    input_df = pandas.DataFrame(input_dict,
                                columns=['datetime', 'data'])    
    x = get_epw_summary_values(input_df,'%Y-%m-%d', 'max', 1)
    y = get_epw_summary_values(input_df, '%Y %m %d', 'mean', 1)
    z = get_epw_summary_values(input_df, '%Y %m %d', 'min', 1)
    assert z.shape == (365, 2)


def test_offset_current_weather_file(epwfile, prismfiles):
    offset_current_weather_file(-123.2,49.2,"Abbotsford",
                                prismfiles,
                                os.path.dirname(epwfile),
                                os.path.dirname(epwfile),
                                epw_filename=os.path.basename(epwfile))
    assert 1 == 1


def test_get_ensemble_averages(alphatas):
    cdfvariable = 'alpha_tas'
    factor = 'monthly'
    lon=-123.015469
    lat=49.249541
    alpha_tas = get_ensemble_averages(cdfvariable=cdfvariable,
                                      lon=lon, lat=lat,
                                      gcm_files=alphatas,
                                      factor=factor,rlen=21)
    assert(len(alpha_tas) == 365)


def test_generate_dry_bulb_temperature(alphatas, deltatas, epwfile):
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    epw_variable_name = 'dry_bulb_temperature'
    with open(epwfile) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_dbt_morph = generate_dry_bulb_temperature(
        epw_data[epw_variable_name],
        epw_data['datetime'],
        lon,lat,
        alphatas,
        deltatas,
        factor,rlen
    )
    assert len(epw_dbt_morph) == len(epw_data[epw_variable_name])


def test_generate_dewpoint_temperature(alphadew,deltadew,epwfile):
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    epw_variable_name = 'dew_point_temperature'
    with open(epwfile) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_dwpt_morph = generate_dewpoint_temperature(
        epw_data[epw_variable_name],
        epw_data['datetime'],
        lon,lat,
        alphadew,
        deltadew,
        factor,rlen
    )
    assert len(epw_dwpt_morph) == len(epw_data[epw_variable_name])


def test_generate_horizontal_radiation(alpharad, epwfile):
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    epw_variable_name = 'global_horizontal_radiation'
    with open(epwfile) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_rad_morph = generate_horizontal_radiation(
        epw_data['global_horizontal_radiation'],
        epw_data['diffuse_horizontal_radiation'],
        epw_data['datetime'],
        lon,lat,
        alpharad,
        factor,rlen
    )
    assert len(epw_rad_morph) == len(epw_data[epw_variable_name])


def test_generate_stretched_series(alphaclt, epwfile):
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    # Alternative variables
    # 'atmospheric_station_pressure'
    # 'direct_normal_radiation'
    # 'relative_humidity'
    # 'wind_speed'
    # 'total_sky_cover'
    epw_variable_name = 'opaque_sky_cover'
    cdfvariable = 'clt'
    morphing_function = morph_opaque_sky_cover

    with open(epwfile) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_var_morph = generate_stretched_series(
        epw_data[epw_variable_name],
        epw_data['datetime'],
        lon,lat,
        cdfvariable,
        alphaclt,
        morphing_function,
        factor, rlen
    )
    assert len(epw_var_morph) == len(epw_data[epw_variable_name])


def test_gen_future_weather_file(alpharhs, prismfiles, epwfile):
    location_name = 'TestSite'
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    prism_files = prismfiles
    epw_variable_name = 'relative_humidity'
    cdfvariable = 'rhs'
    morphing_function = morph_relative_humidity
    epw_read = os.path.dirname(epwfile)
    epw_filename = os.path.basename(epwfile)
    epw_write =  os.path.dirname(epwfile)

    epw_var_morph = gen_future_weather_file(
        location_name=location_name,
        lon=lon,
        lat=lat,
        epw_read=epw_read,
        epw_write=epw_write,
        epw_file_name=epw_filename,
        epw_variable_name=epw_variable_name,
        factor=factor,
        rlen=rlen,
        prism_files=prism_files,
        morphing_climate_files=alpharhs)       
    assert 1 == 1


