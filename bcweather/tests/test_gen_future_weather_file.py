import io
import numpy as np
import pandas
import glob
import pdb
import matplotlib.pyplot as plt 

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

"""
def test_get_climate_data():
    print('Test Read Alpha and Delta Tas')

    rcp = 'rcp85'
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2"] ##, "CNRM-CM5", "CSIRO-Mk3-6-0",
           ##"GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
           ##"MIROC5", "MRI-CGCM3"]
    alpha_files = []
    delta_files = []

    for gcm in gcms:
        print(gcm)
        alpha_file = glob.glob(gcm_dir + 'alpha_tasmax_tasmin_'+ gcm + '_1971-2000_2041-2070.nc')
        delta_file = glob.glob(gcm_dir + 'delta_tas_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
        delta_files.append(delta_file[0])

    print(alpha_files)
    print(delta_files)
    fac = 'monthly'
    tlen = 365
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

    ##reps = [31,28,31,30,31,30,31,31,30,31,30,31]
    ##x = np.repeat(alpha_tas[:,1],reps,axis=0)
    ##print(x)
    print(alpha_tas.shape)
    ##print(delta_tas.shape)
    assert(alpha_tas.shape == (365, len(gcms)))
    ##assert(delta_tas.shape == (12, len(gcms)))
"""


"""
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
"""

"""
def test_generate_dry_bulb_temperature():
    print('Test Read Alpha and Delta Tas')
    lon=-123.015469
    lat=49.249541
    present_range = [1971,2000]
    future_range = [2041,2070]
    factor = 'roll'
    rlen = 21
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
    epw_variable_name = 'dry_bulb_temperature'
    epw_filename = "/storage/data/projects/rci/weather_files/" \
                   + "wx_2016/CAN_BC_Abbotsford.Intl.AP.711080_CWEC2016.epw" 
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_dbt_morph = generate_dry_bulb_temperature(
        epw_data[epw_variable_name],
        epw_data['datetime'],
        lon,lat,
        alpha_files,
        delta_files,
        factor,rlen
    )
    ##x = range(0,len(epw_data[epw_variable_name]))
    ##plt.plot(x,epw_data[epw_variable_name])
    ##plt.plot(x,epw_dbt_morph)
    ##plt.show()
    ##pdb.set_trace()
    assert len(epw_dbt_morph) == len(epw_data[epw_variable_name])


def test_generate_dewpoint_temperature():
    print('Test Read Alpha and Delta Dewpoint')
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    rcp = 'rcp85'
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0",
           "GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
           "MIROC5", "MRI-CGCM3"]
    alpha_files = []
    delta_files = []
    for gcm in gcms:
        print(gcm)
        alpha_file = glob.glob(gcm_dir + 'alpha_dewpoint_'
                              + gcm + '_1971-2000_2041-2070.nc')
        delta_file = glob.glob(gcm_dir + 'delta_dewpoint_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
        delta_files.append(delta_file[0])
    epw_variable_name = 'dew_point_temperature'
    epw_filename = "/storage/data/projects/rci/weather_files/" \
                   + "wx_2016/CAN_BC_Abbotsford.Intl.AP.711080_CWEC2016.epw" 
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_dwpt_morph = generate_dewpoint_temperature(
        epw_data[epw_variable_name],
        epw_data['datetime'],
        lon,lat,
        alpha_files,
        delta_files,
        factor,rlen
    )
    ##x = range(0,len(epw_data[epw_variable_name]))
    ##plt.plot(x,epw_dwpt_morph)
    ##plt.plot(x,epw_data[epw_variable_name])
    ##plt.legend(('Morphed','File'),loc='upper left')
    ##plt.show()
    assert len(epw_dwpt_morph) == len(epw_data[epw_variable_name])

def test_generate_horizontal_radiation():
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    rcp = 'rcp85'
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0",
           "GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
           "MIROC5", "MRI-CGCM3"]
    alpha_files = []
    for gcm in gcms:
        print(gcm)
        alpha_file = glob.glob(gcm_dir + 'alpha_rsds_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
    epw_variable_name = 'global_horizontal_radiation'
    epw_filename = "/storage/data/projects/rci/weather_files/" \
                   + "wx_2016/CAN_BC_Abbotsford.Intl.AP.711080_CWEC2016.epw" 
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_rad_morph = generate_horizontal_radiation(
        epw_data['global_horizontal_radiation'],
        epw_data['diffuse_horizontal_radiation'],
        epw_data['datetime'],
        lon,lat,
        alpha_files,
        factor,rlen
    )
    x = range(0,len(epw_data[epw_variable_name]))

    plt.subplot(2,1,1)
    plt.plot(x,epw_rad_morph[:,0],linewidth=1) # Global
    plt.plot(x,epw_data['global_horizontal_radiation'],
             linewidth=0.5)
    plt.title('Global')
    plt.legend(('Morphed','File'),loc='upper left')
    plt.subplot(2,1,2)
    plt.plot(x,epw_rad_morph[:,1],linewidth=1) # Diffuse
    plt.plot(x,epw_data['diffuse_horizontal_radiation'],
             linewidth=0.5)
    plt.title('Diffuse')
    plt.legend(('Morphed','File'),loc='upper left')
    plt.show()
    ##pdb.set_trace()
    assert len(epw_rad_morph) == len(epw_data[epw_variable_name])


def test_generate_stretched_series():
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    rcp = 'rcp85'
    #'atmospheric_station_pressure'
    #'direct_normal_radiation'
    #'relative_humidity'
    #'wind_speed'
    #'total_sky_cover'
    epw_variable_name = 'opaque_sky_cover'
    cdfvariable = 'clt'
    morphing_function = morph_opaque_sky_cover

    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0",
           "GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
           "MIROC5", "MRI-CGCM3"]
    alpha_files = []
    for gcm in gcms:
        print(gcm)
        alpha_file = glob.glob(gcm_dir + 'alpha_' + cdfvariable + '_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
    epw_filename = "/storage/data/projects/rci/weather_files/" \
                   + "wx_2016/CAN_BC_Abbotsford.Intl.AP.711080_CWEC2016.epw" 
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_var_morph = generate_stretched_series(
        epw_data[epw_variable_name],
        epw_data['datetime'],
        lon,lat,
        cdfvariable,
        alpha_files,
        morphing_function,
        factor,rlen
    )
    x = range(0,len(epw_data[epw_variable_name]))

    plt.plot(x,epw_var_morph)
    plt.plot(x,epw_data[epw_variable_name])
    plt.legend(('Morphed','File'),loc='upper left')
    plt.show()
    ##pdb.set_trace()
    assert len(epw_var_morph) == len(epw_data[epw_variable_name])
"""


def test_gen_future_weather_file():
    location_name = 'TestSite'
    lon=-123.015469
    lat=49.249541
    factor = 'roll'
    rlen = 21
    rcp = 'rcp85'
    prism_files = ['/storage/data/climate/PRISM/dataportal/tmax_monClim_PRISM_historical_run1_198101-201012.nc',
                   '/storage/data/climate/PRISM/dataportal/tmin_monClim_PRISM_historical_run1_198101-201012.nc',
                   '/storage/data/climate/PRISM/dataportal/pr_monClim_PRISM_historical_run1_198101-201012.nc']
    #'atmospheric_station_pressure'
    #'direct_normal_radiation'
    #'relative_humidity'
    #'wind_speed'
    #'total_sky_cover'
    epw_variable_name = 'relative_humidity'
    cdfvariable = 'rhs'
    morphing_function = morph_relative_humidity

    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/bccaq2_tps/epw_factors/"
    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0",
           "GFDL-ESM2G", "HadGEM2-CC", "HadGEM2-ES", "inmcm4",
           "MIROC5", "MRI-CGCM3"]
    alpha_files = []
    for gcm in gcms:
        print(gcm)
        alpha_file = glob.glob(gcm_dir + 'alpha_' + cdfvariable + '_'
                              + gcm + '_1971-2000_2041-2070.nc')
        alpha_files.append(alpha_file[0])
    epw_read = "/storage/data/projects/rci/weather_files/wx_2016/" #\
                   #+ "wx_2016/CAN_BC_Abbotsford.Intl.AP.711080_CWEC2016.epw" 
    epw_write = "/storage/data/projects/rci/weather_files/" 

    epw_var_morph = gen_future_weather_file(
        location_name=location_name,
        lon=lon,
        lat=lat,
        epw_read=epw_read,
        epw_write=epw_write,
        epw_variable_name=epw_variable_name,
        factor=factor,
        rlen=rlen,
        prism_files=prism_files,
        morphing_climate_files=alpha_files)
        
    assert 1 == 1


def test_epw_to_data_frame(epwfile):
    print('EPW to Data Frame')
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns


def test_get_epw_summary_values(epwfile):
    print('Get EPW Summary Values')
    df = epw_to_data_frame(epwfile)
    input_dict = {'datetime': df['datetime'], 'data': df['dry_bulb_temperature']}
    input_df = pandas.DataFrame(input_dict,columns=['datetime', 'data'])
    
    x = get_epw_summary_values(input_df,'%Y-%m-%d', 'max',1)
    y = get_epw_summary_values(input_df, '%Y %m %d', 'mean',1)
    z = get_epw_summary_values(input_df, '%Y %m %d', 'min',1)
    assert z.shape == (365, 2)


def test_offset_current_weather_file():
    offset_current_weather_file(-123.2,49.2,"UVic",
                                ['/storage/data/climate/PRISM/dataportal/tmax_monClim_PRISM_historical_run1_198101-201012.nc',
                                 '/storage/data/climate/PRISM/dataportal/tmin_monClim_PRISM_historical_run1_198101-201012.nc',
                                 '/storage/data/climate/PRISM/dataportal/pr_monClim_PRISM_historical_run1_198101-201012.nc'],
                                "/storage/data/projects/rci/weather_files/wx_files/",
                                "/storage/data/projects/rci/weather_files/wx_files/morphed_files/")
    assert 1 == 1


