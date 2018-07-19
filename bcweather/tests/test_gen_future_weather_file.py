import io
import numpy
import pandas
import glob
from datetime import datetime
from netCDF4 import Dataset

from bcweather import get_epw_header, get_climate_data, get_ensemble_averages
from bcweather import get_epw_summary_values, gen_prism_offset_weather_file, generate_dry_bulb_temperature, morph_dry_bulb_temperature
from bcweather import generate_dewpoint_temperature, morph_dewpoint_temperature, generate_horizontal_radiation, morph_horizontal_radiation
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
def test_get_climate_data(ncfile):
    print(' ')
    print('Test get climate data')
    gcm = 'ACCESS1-0'
    gcm_file = "/storage/data/climate/downscale/CMIP5/building_code/"+gcm+"/dewpoint_day_"+gcm+"_historical+rcp85_r1i1p1_19500101-21001231.nc"
    ##"/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/"+gcm+"/tasmax_day_BCCAQ2_"+gcm+"_rcp85_r1i1p1_1951-2000.nc" 
    with Dataset(gcm_file) as f:
        past_climate = get_climate_data(f, 
                                        49.03,-122.36,  'dewpoint', [1971, 2000],'%m')
    print('Get climate present')
    print(past_climate['mean'])

    gcm_file = "/storage/data/climate/downscale/CMIP5/building_code/"+gcm+"/dewpoint_day_"+gcm+"_historical+rcp85_r1i1p1_19500101-21001231.nc"
    ###"/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/"+gcm+"/tasmax_day_BCCAQ2_"+gcm+"_rcp85_r1i1p1_2001-2100.nc" 
    with Dataset(gcm_file) as f:
        proj_climate = get_climate_data(f, 
                                        49.03,-122.36,  'dewpoint', [2041, 2070],'%m')
    print('Get climate future')
    print(proj_climate['mean'])
    print('Delta')
    print(proj_climate['mean']-past_climate['mean'])
    assert len(past_climate['mean']) == 12
"""

"""
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

    print('Past List of Files')
    print(past_files)
    print('Future List of Files')
    print(proj_files)
    factor = '%m'
    tasmax_present = get_ensemble_averages(cdfvariable=cdfvariable,
                                           lon=-122.36,lat=49.03,
                                           gcm_files=past_files,
                                           time_range=[1971,2000],
                                           factor=factor)
    past_mean = tasmax_present['std']
    tasmax_future = get_ensemble_averages(cdfvariable=cdfvariable,
                                          lon=-122.36,lat=49.03,
                                          gcm_files=proj_files,
                                          time_range=[2041,2070],
                                          factor=factor)
    proj_mean = tasmax_future['std']    
    print('Past means')
    print(past_mean)
    print('Proj means')
    print(proj_mean)   
    print('Deltas')
    print(proj_mean-past_mean)
    print('Ens')
    print(numpy.mean(proj_mean/past_mean,axis=1))
    assert(proj_mean.shape == (12,10))
"""


"""
def test_epw_to_data_frame(epwfile):
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns
"""

"""
def test_get_epw_summary_values(epwfile):
    df = epw_to_data_frame(epwfile)
    epw_tas = numpy.array(df['dry_bulb_temperature'])
    epw_dates = df['datetime']
    print(epw_dates)
    epw_months = pandas.DatetimeIndex(epw_dates).month
    print('EPW Months')
    print(epw_months[0:30])
    print('Match')
    ix = epw_months == 1
    print(epw_tas[ix])
    print(len(epw_tas[ix]))
    y = get_epw_summary_values(df['dry_bulb_temperature'], df['datetime'],
                               '%m','mean')
    print(y)
    assert y.shape == (12, 3)
"""

# def test_gen_prism_offset_weather_file():
#    gen_prism_offset_weather_file(49.2,-123.2)
#    assert 1 == 1

##def test_gen_future_weather_file():
##    print('Test future weather file production')
##    gcms = ["ACCESS1-0", "CanESM2", "CNRM-CM5", "CSIRO-Mk3-6-0", "GFDL-ESM2G",
##            "HadGEM2-CC", "HadGEM2-ES", "inmcm4", "MIROC5", "MRI-CGCM3"]
##    x = gen_future_weather_file(epw_filename="/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw",
##                                epw_output_filename="/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/future_test.epw",
##                                present_range=[1971, 2000],
##                                future_range=[2041, 2070],
##                                gcms=gcms)
##    assert type(x) == str

# def test_stretch_gcm_variable(epwfile):
#    print('Test GCM stretch with GCM ensemble')
#    df = epw_to_data_frame(epwfile)
#    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
#    x = stretch_gcm_variable(epw_data=df,
#                             lon=-122.36,lat=49.03,
#                             epw_variable="relative_humidity",
#                             netcdf_variable="psl",
#                             gcms=gcms,
#                             present_range=[1971,2000],
#                             future_range=[2041,2070])
#    assert type(x) == pandas.DataFrame

"""
def test_generate_dry_bulb_temperature(epwfile):
    print('Test GCM stretch with GCM ensemble')
    rcp = 'rcp85'
    df = epw_to_data_frame(epwfile)
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/" 
    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
    glen = len(gcms)    
    tasmax_present_gcm_files = []
    tasmax_future_gcm_files = []
    tasmin_present_gcm_files = []
    tasmin_future_gcm_files = []

    for gcm in gcms:
        tasmax_present_file = glob.glob(gcm_dir + gcm + '/' + 'tasmax' + '_day_*' + rcp + '*1951-2000.nc')
        tasmax_future_file = glob.glob(gcm_dir + gcm + '/' + 'tasmax' + '_day_*' + rcp + '*2001-2100.nc')
        tasmax_present_gcm_files.append(tasmax_present_file[0])
        tasmax_future_gcm_files.append(tasmax_future_file[0])
        tasmin_present_file = glob.glob(gcm_dir + gcm + '/' + 'tasmin' + '_day_*' + rcp + '*1951-2000.nc')
        tasmin_future_file = glob.glob(gcm_dir + gcm + '/' + 'tasmin' + '_day_*' + rcp + '*2001-2100.nc')
        tasmin_present_gcm_files.append(tasmin_present_file[0])
        tasmin_future_gcm_files.append(tasmin_future_file[0])
        
    test = generate_dry_bulb_temperature(epw_tas=df['dry_bulb_temperature'],
                                         epw_dates=df['datetime'],
                                         lon=-122.36, lat=49.03,
                                         tasmax_present_gcm_files=tasmax_present_gcm_files,
                                         tasmax_future_gcm_files=tasmax_future_gcm_files,
                                         tasmin_present_gcm_files=tasmin_present_gcm_files,
                                         tasmin_future_gcm_files=tasmin_future_gcm_files,
                                         present_range=[1971,2000],
                                         future_range=[2041,2070],
                                         factor='%m'
                                        )
    assert type(test) == numpy.array
"""

"""
def test_generate_dewpoint_temperature(epwfile):
    rcp = 'rcp85'
    df = epw_to_data_frame(epwfile)
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/" 
    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
    glen = len(gcms)    
    dewpoint_present_gcm_files = []
    dewpoint_future_gcm_files = []

    for gcm in gcms:
        dewpoint_present_file = glob.glob(gcm_dir + gcm + '/' + 'dewpoint' + '_day_*' + rcp + '*19500101-21001231.nc')
        dewpoint_future_file = glob.glob(gcm_dir + gcm + '/' + 'dewpoint' + '_day_*' + rcp + '*19500101-21001231.nc')
        dewpoint_present_gcm_files.append(dewpoint_present_file[0])
        dewpoint_future_gcm_files.append(dewpoint_future_file[0])
        
    print(dewpoint_present_gcm_files)
    test = generate_dewpoint_temperature(epw_dwpt=df['dew_point_temperature'],
                                         epw_dates=df['datetime'],
                                         lon=-122.36, lat=49.03,
                                         present_gcm_files=dewpoint_present_gcm_files,
                                         future_gcm_files=dewpoint_future_gcm_files,
                                         present_range=[1971,2000],
                                         future_range=[2041,2070],
                                         factor='%m',
                                        )
    assert len(test) == len(df['dew_point_temperature'])
"""

def test_generate_horizontal_radiation(epwfile):
    rcp = 'rcp85'
    df = epw_to_data_frame(epwfile)
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/" 
    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
    glen = len(gcms)    
    rsds_present_gcm_files = []
    rsds_future_gcm_files = []

    for gcm in gcms:
        rsds_present_file = glob.glob(gcm_dir + gcm + '/' + 'rsds' + '_day_*' + rcp + '*19500101-21001231.nc')
        rsds_future_file = glob.glob(gcm_dir + gcm + '/' + 'rsds' + '_day_*' + rcp + '*19500101-21001231.nc')
        rsds_present_gcm_files.append(rsds_present_file[0])
        rsds_future_gcm_files.append(rsds_future_file[0])
        
    print(rsds_present_gcm_files)
    test = generate_horizontal_radiation(epw_ghr=df['global_horizontal_radiation'],
                                         epw_dhr=df['diffuse_horizontal_radiation'],
                                         epw_dates=df['datetime'],
                                         lon=-122.36, lat=49.03,
                                         present_gcm_files=rsds_present_gcm_files,
                                         future_gcm_files=rsds_future_gcm_files,
                                         present_range=[1971,2000],
                                         future_range=[2041,2070],
                                         factor='%m',
                                        )
    print(numpy.array(df['global_horizontal_radiation'][0:31]))
    print(numpy.array(df['diffuse_horizontal_radiation'][0:31]))
    print(test[0:31,:])
    assert len(test[:,0]) == len(df['global_horizontal_radiation'])

"""
def test_generate_stretched_series(epwfile):
    rcp = 'rcp85'
    df = epw_to_data_frame(epwfile)
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/" 
    gcms = ["ACCESS1-0","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G","HadGEM2-CC","HadGEM2-ES","inmcm4","MIROC5","MRI-CGCM3"]
    glen = len(gcms)    
    rsds_present_gcm_files = []
    rsds_future_gcm_files = []

    for gcm in gcms:
        rsds_present_file = glob.glob(gcm_dir + gcm + '/' + 'rsds' + '_day_*' + rcp + '*19500101-21001231.nc')
        rsds_future_file = glob.glob(gcm_dir + gcm + '/' + 'rsds' + '_day_*' + rcp + '*19500101-21001231.nc')
        rsds_present_gcm_files.append(rsds_present_file[0])
        rsds_future_gcm_files.append(rsds_future_file[0])
        
    print(rsds_present_gcm_files)
    test = generate_stretched_series(epw_data=df['direct_normal_radiation'],
                                     epw_dates=df['datetime'],
                                     lon=-122.36, lat=49.03,
                                     cdfvariable='rsds',
                                     present_gcm_files=rsds_present_gcm_files,
                                     future_gcm_files=rsds_future_gcm_files,
                                     present_range=[1971,2000],
                                     future_range=[2041,2070],
                                     factor='%m',
                                     morphing_function=morph_direct_normal_radiation
                                    )
    print(numpy.array(df['direct_normal_radiation'][0:31]))
    print(test[0:31])
    assert len(test) == len(df['direct_normal_radiation'])
"""
