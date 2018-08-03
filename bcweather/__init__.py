from datetime import datetime
import re

import numpy as np
from netCDF4 import Dataset
import netCDF4 as cdf
from typing import IO
import pandas
import glob

from .epw import epw_to_data_frame



#-----------------------------------
# Core Morphing Functions

# Take the daily/monthly delta factors, compute the morphed series
# and return the new series
def morph_dry_bulb_temperature(epw_tas: pandas.Series,
                               epw_dates: pandas.Series,
                               tasmax_delta: list,
                               tasmin_delta: list,
                               tas_delta: list,
                               factor: str
                               ):
    """ morph_dry_bulb_temperature(pandas.Series,pandas.Series,
                                   list,list,list,
                                   str)

        This function takes in hourly temperature data from the epw file,
        the delta factors for max, min and mean temperature, and returns 
        the "morphed" future dry bulb temperature.

        Args:
            epw_tas(Series): A hourly temperature column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            *_delta(list): Lists of change factors for max,min,mean temperature.
            factor(str): Date factor with which to average.

        Returns:
            an array with future dry bulb temperature data.
    """

    # ----------------------------------------------------------
    # Time averaged temperatures based on specific factor
    epw_daily_averages = get_epw_summary_values(epw_tas,
                                                epw_dates,'%Y %m %d','mean')
    epw_tas_averages = get_epw_summary_values(epw_daily_averages['data'],
                                              epw_daily_averages['datetime'],factor,'mean')['data']
    epw_daily_max = get_epw_summary_values(epw_tas,
                                           epw_dates,'%Y %m %d','max')
    epw_tas_max = get_epw_summary_values(epw_daily_max['data'],
                                         epw_daily_max['datetime'],factor,'mean')['data']
    epw_daily_min = get_epw_summary_values(epw_tas,
                                           epw_dates,'%Y %m %d','min')
    epw_tas_min = get_epw_summary_values(epw_daily_min['data'],
                                         epw_daily_min['datetime'],factor,'mean')['data']
        
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed_dbt = np.zeros(len(epw_tas))
    for uf in unique_factors:
        ix = epw_factor == uf
        shift = epw_tas[ix] + tas_delta[uf-1]
        alpha = (tasmax_delta[uf-1] - tasmin_delta[uf-1]) / \
            (epw_tas_max[uf-1] - epw_tas_min[uf-1])
        anoms = epw_tas[ix] - epw_tas_averages[uf-1]
        morphed_dbt[ix] = round(shift + alpha * anoms,1)
    return(morphed_dbt)

# ------------------------------------------------------
def morph_dewpoint_temperature(epw_dwpt: pandas.Series,
                               epw_dates: pandas.Series,
                               dewpoint_delta: list,
                               dewpoint_alpha: list,
                               factor: str
                               ):
    """ morph_dewpoint_temperature(pandas.Series,pandas.Series,
                                   list,list,
                                   str)

        This function takes in hourly dewpoint temperature data from the epw file,
        the delta and alpha factors for dewpoint temperature, and returns 
        the "morphed" future dewpoint temperature.

        Args:
            epw_dwpt(Series): A hourly dewpoint temperature column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            dewpoint_delta(list): Lists of shift factors for dewpoint temperature.
            dewpoint_alpha(list): Lists of stretch factors for dewpoint temperature.
            factor(str): Date factor with which to average.

        Returns:
            an array with future dry bulb temperature data.
    """
    # Time averaged temperatures based on specific factor
    epw_daily_averages = get_epw_summary_values(epw_dwpt,
                                                epw_dates,'%Y %m %d','mean')
    epw_averages = get_epw_summary_values(epw_daily_averages['data'],
                                              epw_daily_averages['datetime'],factor,'mean')['data']
    # 
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed_dwpt = np.zeros(len(epw_dwpt))

    ix = epw_factor == unique_factors[0]
    for uf in unique_factors:
        ix = epw_factor == uf
        shift = epw_averages[uf-1] + dewpoint_delta[uf-1]
        stretch = (epw_dwpt[ix] - epw_averages[uf-1]) * dewpoint_alpha[uf-1]
        morphed_dwpt[ix] = round(shift + stretch,1)
    return(morphed_dwpt)


# ------------------------------------------------------
def morph_direct_normal_radiation(epw_dnr: pandas.Series,
                                  epw_dates: pandas.Series,
                                  alpha: list,
                                  factor: str
                                 ):
    """ morph_direct_normal_radiation(pandas.Series,pandas.Series,
                                      list,str)

        This function takes in hourly direct normal radiation data,
        the alpha factors for stretching, and returns the "morphed" 
        future data. This is uses cloud cover to stretch (inversely
        proportional to cloud cover changes).

        Args:
            epw_dnr(Series): An hourly data column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            alpha(list): List of cloud cover stretch factors.
            factor(str): Date factor with which to average.

        Returns:
            an array with future hourly epw data.
    """
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed = np.zeros(len(epw_dnr))

    for uf in unique_factors:
        ix = epw_factor == uf
        stretch = alpha[uf-1]
        morphed[ix] = round(epw_dnr[ix] / stretch,0)
    return(morphed)

# ------------------------------------------------------
def morph_horizontal_radiation(epw_ghr: pandas.Series,
                               epw_dhr: pandas.Series,
                               epw_dates: pandas.Series,
                               alpha: list,
                               factor: str
                              ): 
    """ morph_horizontal_radiation(pandas.Series,pandas.Series,pandas.Series,
                                      list,str)

        This function takes in both global horizontal radiation and 
        diffuse horizontal radiation, the alpha factors for stretching,
        and returns the "morphed" future data. The two horizontal radiation
        series are morphed simultaneously to preserve the relative relationship
        between the two variables (Global = Direct Horiz. + Diffuse Horiz.). 

        Args:
            epw_ghr(Series): An hourly data column from the epw file.
            epw_dhr(Series): An hourly data column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            alpha(list): List of radiation stretch factors.
            factor(str): Date factor with which to average.

        Returns:
            an 2D array with future hourly epw data.
    """

    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed_rad = np.zeros((len(epw_ghr),2))

    for uf in unique_factors:
        ix = epw_factor == uf
        global_hz_rad = epw_ghr[ix]
        flag = global_hz_rad == 0
        diffuse_hz_rad = epw_dhr[ix]
        diffuse_to_total_ratio = diffuse_hz_rad / global_hz_rad
        diffuse_to_total_ratio[flag] = 0
        morphed_global_hz_rad = global_hz_rad * alpha[uf-1]
        morphed_diffuse_hz_rad = morphed_global_hz_rad * diffuse_to_total_ratio
        morphed_rad[ix,0] = round(morphed_global_hz_rad,0)
        morphed_rad[ix,1] = round(morphed_diffuse_hz_rad,0)
    return(morphed_rad)

# ------------------------------------------------------
def morph_by_stretch(epw_data: pandas.Series,
                     epw_dates: pandas.Series,
                     alpha: list,
                     factor: str
                    ):
    """ morph_by_stretch(pandas.Series,pandas.Series,
                         list,str)

        This function takes in hourly weather file data,
        the alpha factors for stretching, and returns the "morphed" 
        future data. This ise used by multiple epw variables.

        Args:
            epw_data(Series): An hourly data column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            alpha(list): List of stretch factors.
            factor(str): Date factor with which to average.

        Returns:
            an array with future hourly epw data.
    """
    # FIX - replace with specified time factor averaging (like DBT)
    epw_averages = get_epw_summary_values(epw_data,
                                          epw_dates,factor,'mean')
 
    # FIX to replace with alternative to if statement if possible
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed = np.zeros(len(epw_data))

    for uf in unique_factors:
        ix = epw_factor == uf
        stretch = alpha[uf-1]
        morphed[ix] = epw_data[ix] * stretch
    return(morphed)

# ------------------------------------------------------
def morph_relative_humidity(epw_rhs: pandas.Series,
                            epw_dates: pandas.Series,
                            rhs_alpha: list,
                            factor: str
                           ):
    morphed_rhs = morph_by_stretch(epw_rhs,epw_dates,
                                   rhs_alpha,factor)    
    morphed_rhs[morphed_rhs > 100] = 100
    rv = np.asarray(morphed_rhs).astype(int)
    return(np.round(rv,0))

# ------------------------------------------------------
def morph_atmospheric_station_pressure(epw_psl: pandas.Series,
                                       epw_dates: pandas.Series,
                                       psl_alpha: list,
                                       factor: str
                                      ):
    morphed_psl = morph_by_stretch(epw_psl,epw_dates,
                                   psl_alpha,factor)
    rv = np.asarray(morphed_psl).astype(int)
    return(np.round(rv,0))
 
# ------------------------------------------------------
def morph_windspeed(epw_wspd: pandas.Series,
                    epw_dates: pandas.Series,
                    wspd_alpha: list,
                    factor: str
                   ):
    morphed_wspd = morph_by_stretch(epw_wspd,epw_dates,
                                   wspd_alpha,factor)    
    rv = np.asarray(morphed_wspd).astype(int)
    return(np.round(rv,1))

# ------------------------------------------------------

def morph_total_sky_cover(epw_tsc: pandas.Series,
                          epw_dates: pandas.Series,
                          tsc_alpha: list,
                          factor: str
                         ):
    morphed_tsc = morph_by_stretch(epw_tsc,epw_dates,
                                   tsc_alpha,factor)    
    morphed_tsc[morphed_tsc > 10] = 10    
    rv = np.asarray(morphed_tsc).astype(int)
    return(np.round(rv,0))

# ------------------------------------------------------

def morph_opaque_sky_cover(epw_osc: pandas.Series,
                           epw_dates: pandas.Series,
                           osc_alpha: list,
                           factor: str
                         ):
    morphed_osc = morph_by_stretch(epw_osc,epw_dates,
                                   osc_alpha,factor)    
    morphed_osc[morphed_osc > 10] = 10    
    rv = np.asarray(morphed_osc).astype(int)
    return(np.round(rv,0))

# ------------------------------------------------------

def morph_liquid_precip_quantity(epw_pr: pandas.Series,
                                 epw_dates: pandas.Series,
                                 osc_alpha: list,
                                 factor: str
                                ):
    morphed_pr = morph_by_stretch(epw_pr,epw_dates,
                                  pr_alpha,factor)    
    rv = np.asarray(morphed_pr).astype(int)
    return(np.round(rv,0))

# ------------------------------------------------------

# Unlikely to be modified due to data contraints
# def morph_wind_direction():
# def morph_snow_depth():
# def morph_liquid_precip_depth():
# def morph_precip_quantity():

# -----------------------------------------------------------------

def generate_dry_bulb_temperature(epw_tas: pandas.Series,
                                  epw_dates: pandas.Series,
                                  lon: float, lat: float,
                                  tasmax_present_climate_files: list,
                                  tasmax_future_climate_files: list,
                                  tasmin_present_climate_files: list,
                                  tasmin_future_climate_files: list,
                                  present_range: range,
                                  future_range: range,
                                  factor: str
                                 ):
    """ generate_dry_bulb_temperature(pandas.Series,pandas.Series,
                                      float,float,
                                      list,list,list,list,
                                      range,range,str)

        This function takes in data from the epw file, and returns 
        the dataframe with the dry bulb temperature column having been
        replaced with future ("morphed") versions of dry bulb temperature.

        Args:
            epw_tas(Series): A hourly temperature column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            *_gcm_files(list): The list of gcm files for past and future 
            tasmax and tasmin
            present_range(range): Year bounds for the present climate.
            future_range(range): Year bounds for the future climate.
            factor(str): Date factor with which to average.

        Returns:
            a list with future dry bulb temperature data.
    """
    # Morphing factors from the input gcm files
    tasmax_present = get_ensemble_averages(cdfvariable='tasmax',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmax_present_climate_files,
                                           time_range=present_range,
                                           factor=factor)
    tasmax_future = get_ensemble_averages(cdfvariable='tasmax',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmax_future_climate_files,
                                           time_range=future_range,
                                           factor=factor)
    tasmax_delta = tasmax_future['mean'] - tasmax_present['mean']
    
    tasmin_present = get_ensemble_averages(cdfvariable='tasmin',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmin_present_climate_files,
                                           time_range=present_range,
                                           factor=factor)

    tasmin_future = get_ensemble_averages(cdfvariable='tasmin',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmin_future_climate_files,
                                           time_range=future_range,
                                           factor=factor)
    tasmin_delta = tasmin_future['mean'] - tasmin_present['mean']

    # One could also to use change in variability instead of change in diurnal cycle
    # (as is done in the "stretch" definition for other variables in Belcher).
    tas_delta = (tasmax_future['mean']+tasmin_future['mean'])/2 - (
        tasmax_present['mean']+tasmin_present['mean'])/2
    morphed_epw_tas = morph_dry_bulb_temperature(epw_tas,epw_dates,
                                                 np.nanmean(tasmax_delta,axis=1),
                                                 np.nanmean(tasmin_delta,axis=1),
                                                 np.nanmean(tas_delta,axis=1),
                                                 factor)
    return morphed_epw_tas
# -----------------------------------------------------------------

def generate_dewpoint_temperature(epw_dwpt: pandas.Series,
                                  epw_dates: pandas.Series,
                                  lon: float, lat: float,
                                  present_climate_files: list,
                                  future_climate_files: list,
                                  present_range: range,
                                  future_range: range,
                                  factor: str
                                 ):
    """ generate_dewpoint_temperature(pandas.Series, pandas.Series,
                                      float,float,
                                      list,list,range,range,str)

        This function takes in dewpoint data from the epw file, and returns 
        the future ("morphed") versions of dewpoint temperature.

        Args:
            data(Series): Dewpoint series from EPW file
            dates(Series): Dates from EPW file
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            present_climate_files(list):
            future_climate_files(list):
            present_range(range): Year bounds for the present climate.
            future_range(range): Year bounds for the future climate.
            factor(str): Averaging factor
        Returns:
            a numpy array of future dewpoint temperature data.
    """
    dewpoint_present = get_ensemble_averages(cdfvariable='dewpoint',
                                             lon=lon, lat=lat,
                                             gcm_files=present_climate_files,
                                             time_range=present_range,
                                             factor=factor)
    dewpoint_future = get_ensemble_averages(cdfvariable='dewpoint',
                                             lon=lon, lat=lat,
                                             gcm_files=future_climate_files,
                                             time_range=future_range,
                                             factor=factor)
    dewpoint_delta = dewpoint_future['mean'] - dewpoint_present['mean']
    dewpoint_alpha = dewpoint_future['std'] / dewpoint_present['std']
    morphed_dwpt = morph_dewpoint_temperature(epw_dwpt,epw_dates,
                                              np.nanmean(dewpoint_delta,axis=1),
                                              np.nanmean(dewpoint_alpha,axis=1),
                                              factor)    
    return morphed_dwpt
# -----------------------------------------------------------------

def generate_horizontal_radiation(epw_ghr: pandas.Series,
                                  epw_dhr: pandas.Series,
                                  epw_dates: pandas.Series,
                                  lon: float, lat: float,
                                  present_climate_files: list,
                                  future_climate_files: list,
                                  present_range: range,
                                  future_range: range,
                                  factor: str
                                 ):
    """ generate_horizontal_radiation(pandas.Series, pandas.Series,pandas.Series
                                      float,float,
                                      list,list,range,range,str)

        This function takes in global and diffuse horizontal radiation data from the epw file, 
        and returns the morphed versions.

        Args:
            data(Series): Global Horizontal Radiation series from EPW file
            data(Series): Diffuse Horizontal Radiation series from EPW file
            dates(Series): Dates from EPW file
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            present_climate_files(list): Uses RSDS 
            future_climate_files(list): RSDS 
            present_range(range): Year bounds for the present climate.
            future_range(range): Year bounds for the future climate.
            factor(str): Averaging factor
        Returns:
            a 2D numpy array of future horizontal radiation data.
    """

    rsds_present = get_ensemble_averages(cdfvariable='rsds',
                                         lon=lon, lat=lat,
                                         gcm_files=present_climate_files,
                                         time_range=present_range,
                                         factor=factor)

    rsds_future = get_ensemble_averages(cdfvariable='rsds',
                                             lon=lon, lat=lat,
                                             gcm_files=future_climate_files,
                                             time_range=future_range,
                                             factor=factor)
    rsds_alpha = rsds_future['std'] / rsds_present['std']
    morphed_horiz_rad = morph_horizontal_radiation(epw_ghr,epw_dhr,epw_dates,
                                                   np.nanmean(rsds_alpha,axis=1),
                                                   factor)
    return morphed_horiz_rad
#----------------------------------------------------------------

def generate_stretched_series(epw_data: pandas.Series,
                              epw_dates: pandas.Series,
                              lon: float, lat: float,
                              cdfvariable: str,
                              present_climate_files: list,
                              future_climate_files: list,
                              present_range: range,
                              future_range: range,
                              morphing_function,
                              factor: str
                             ):
    """ generate_stretched_series(pandas.Series, pandas.Series,
                               float,float,str,
                               list,list,range,range,str)

        This function takes in a single series of data from the epw file, 
        and returns the morphed version.

        Args:
            data(Series): EPW Series
            dates(Series): Dates from EPW file
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            cdfvariable(str): Netcdf variable
            present_climate_files(list): 
            future_climate_files(list): 
            present_range(range): Year bounds for the present climate.
            future_range(range): Year bounds for the future climate.
            factor(str): Averaging factor
            morphing_function: Function to produce the stretched series
        Returns:
            a numpy array of future data.
    """

    gcm_present = get_ensemble_averages(cdfvariable=cdfvariable,
                                        lon=lon, lat=lat,
                                        gcm_files=present_climate_files,
                                        time_range=present_range,
                                        factor=factor)

    gcm_future = get_ensemble_averages(cdfvariable=cdfvariable,
                                       lon=lon, lat=lat,
                                       gcm_files=future_climate_files,
                                       time_range=future_range,
                                       factor=factor)
    alpha = gcm_future['std'] / gcm_present['std']
    morphed_epw = morphing_function(epw_data,epw_dates,
                                    np.nanmean(alpha,axis=1),
                                    factor)
    return morphed_epw


#----------------------------------------------------------------
#----------------------------------------------------------------

def get_ensemble_averages(cdfvariable: str,
                          lon: float,
                          lat: float,
                          gcm_files: list,
                          time_range: range,
                          factor: str,
                         ):
    """ get_ensemble_averages(cdfvariable,lat,lon,gcms_files,time_range,factor)

        Returns the climatological averages of the specified netcdf files.
        Args:
            cdfvariable(str): The variable to read from the netcdf file.
            lon(float): The longitude to read data from.
            lat(float): The latitude to read data from.
            gcm_files(list): Ensemble of GCMs to use
            time_range(range): The range of years to read data from, to.
            factor(str): The time interval over which to average.
        Returns:
            a dict with two numpy arrays. One for each time interval of the year.
    """
    # Only options for monthly or daily averaging
    tlen = 0
    if factor == '%m':
        tlen = 12
    if factor == '%m %d':
        tlen = 365
    # Compute the climatologies
    mean_aggregate = np.zeros((tlen, len(gcm_files)))
    std_aggregate = np.zeros((tlen, len(gcm_files)))
    for i, gcm_file in enumerate(gcm_files):
        with Dataset(gcm_file) as f:
            file_climate = get_climate_data(
                f, lat, lon, cdfvariable, time_range,factor)

        mean_aggregate[:, i] = file_climate['mean'][:, 0]
        std_aggregate[:, i] = file_climate['std'][:, 0]

    ens_climatologies =  {'mean':mean_aggregate,
                          'std':std_aggregate}
    return ens_climatologies

# Address the netcdf time calendar issues
def cftime_to_datetime(data_dates,calendar):
    # Convert cftime dates to string 
    ex_years = [date.strftime('%Y') for date in data_dates]
    ex_months = [date.strftime('%m') for date in data_dates]
    ex_days = [date.strftime('%d') for date in data_dates]
    # Concatenate to full date
    ymd = [x+'-'+y+'-'+z for x,y,z in zip(ex_years,ex_months,ex_days)]
    #Set irregular February dates to Feb 28 temporarily
    if (calendar=='360_day'):
        ymd = [re.sub('-02-29','-02-28',x) for x in ymd]
        ymd = [re.sub('-02-30','-02-28',x) for x in ymd]
    # Convert to datetime object
    dates_list = [datetime.strptime(date, '%Y-%m-%d') for date in ymd]
    # Convert to array
    dates_array = np.asarray(dates_list)
    return(dates_array)

# Fill in missing leap and month dates for 365/360 calendars
def format_netcdf_series(time_range,calendar,data):
    # Set up full date series 
    startyear,endyear = time_range
    full_dates = pandas.date_range(str(startyear)+'-01-01',str(endyear)+'-12-31')
    ex_years = [date.strftime('%Y') for date in full_dates]
    ex_months = [date.strftime('%m') for date in full_dates]
    ex_days = [date.strftime('%d') for date in full_dates]
    dates_matrix = np.column_stack((ex_years,ex_months,ex_days))
    ymd = [x+'-'+y+'-'+z for x,y,z in zip(ex_years,ex_months,ex_days)]
    dates_list = [datetime.strptime(date, '%Y-%m-%d') for date in ymd]
    ylen = int(np.diff(time_range))+1
    empty_vector = np.empty(ylen*365)
    empty_vector[:] = np.nan
 
    # Add NAN for missing days where needed
    # Distribute missing days equally through 360 day calendar
    if calendar == '360_day':
        hix = np.repeat(True,365*ylen)
        blanks = np.arange(72,365*ylen,73)
        hix[blanks] = False
        empty_vector[hix] = data        
        data = empty_vector

    # If 360 it must still pass through the 365 option to add the leap day
    if calendar == '365_day' or calendar == '360_day':
        indices = ['-02-29' in s for s in ymd]        
        empty_vector = np.empty(len(ymd))
        empty_vector[:] = np.nan
        empty_vector[~np.array(indices)] = data        
        dates_matrix = {'Time':dates_list,'Data':empty_vector}

    # If gregorian or standard it should have necessary dates
    if calendar != '365_day' and calendar != '360_day':
        dates_matrix = {'Time':dates_list,'Data':data}

    return(dates_matrix)    

def get_climate_data(nc: Dataset,
                     lat: float,
                     lon: float,
                     cdfvariable: str,
                     time_range: list,
                     factor: str,
                     ):
    """ get_climate_data(Dataset, float, float, list,str)

        Gets a list of data for each day of each year within time_range from
        the climate file where the location is closest to lat, lon.

        Args:
            nc(Dataset): An open netCDF4.Dataset object.
            lat(float): The latitude to read data from.
            lon(float): The longitude to read data from.
            cdfvariable(str): The variable to read from the netcdf file.
            time_range(range): The range of years to read data from, to.
            factor(str): The time factor with which to average.
        Returns:
            a dict with two lists. 
            The first list contains the mean values, while the second
            list contains the standard deviations.
    """

    # Get a list of the dates in the climate file.
    data_dates = cdf.num2date(nc["time"][:], nc["time"].units,nc["time"].calendar)
    dates_array = cftime_to_datetime(data_dates,nc['time'].calendar)
    startyear,endyear = time_range

    t0 = np.argwhere(dates_array >= datetime(startyear, 1, 1)).min()
    if nc['time'].calendar == "360_day":
        tn = np.argwhere(dates_array <= datetime(endyear, 12, 30)).max()
    else:
        tn = np.argwhere(dates_array <= datetime(endyear, 12, 31)).max()

    # Get the latitude of each location in the file.
    lat_data = nc.variables["lat"][:]
    # Get the logitude of each location in the file.
    lon_data = nc.variables["lon"][:]

    # Find the indices of the data with the closest lat and lon to
    # those passed.
    lat_index = np.absolute(lat_data - lat).argmin()
    lon_index = np.absolute(lon_data - lon).argmin()

    # Grab the actual relevant data from the file (tn+1 to fix the indexing)
    data = nc.variables[cdfvariable][t0:tn+1, lat_index, lon_index]
    standard_data = format_netcdf_series(time_range,nc["time"].calendar,data)
    data_frame = pandas.DataFrame(standard_data, columns=['Time', 'Data'])
    time_mean = data_frame.groupby(
        data_frame['Time'].dt.strftime(factor)).mean().values
    time_std = data_frame.groupby(
        data_frame['Time'].dt.strftime(factor)).std().values
    # Remove the extra date that only appears with leap years
    if factor == '%m %d':
        time_mean = time_mean[0:365]
        time_std = time_std[0:365]
    data_clim = {'mean':time_mean, 'std':time_std}
    return data_clim

#----------------------------------------------------------------
#----------------------------------------------------------------

def check_epw_variable_name(epw_variable_name: str):
    from .epw import field_names
    if epw_variable_name not in field_names:
        print(epw_variable_name+' is not an EPW variable.')
        raise SystemExit
    else:
        print(epw_variable_name+' is an EPW variable')

def check_epw_inputs(epw_filename: str,                     
                     lon: float, lat: float,
                     epw_output_filename: str):

    if epw_filename == None and (lon == None or lat == None):
        print('No EPW inputs have been provided.')
        print('Either an EPW file or lon/lat coordinates are required.')
        raise SystemExit

    if epw_filename != None and (lon != None or lat != None):
        print('Both and EPW File and location coordinates included.')
        print('Only the EPW file location will be used for morphing.')

    if 'epw' not in epw_output_filename:
        print('Output file must be an EPW file')
        raise SystemExit
    
def gen_future_weather_file(epw_output_filename: str,
                            epw_variable_name: str,
                            present_range: range,
                            future_range: range,
                            present_climate_files: list,
                            future_climate_files: list,
                            factor: str,
                            epw_filename=None,
                            lon=None, lat=None):
    """ gen_future_weather_file(str, float, float, str, str, range, range, list, list)

        Regenerates the passed epw file into a weather file represeting future
        data.

        Args:
            epw_filename(str): The path to the epw file to regenerate. If not provided
                               lon and lat coordinates must be provided.
            lon (float): EPW Location coordinate
            lat (float): EPW Location coordinate
            epw_output_filename(str): The path to the future epw file to create
            present_range(range): The range of years that makes up "present"
                          for this particular run.
            future_range(range): The range of years that makes up "future" for
                         this particular run.
            gcm(list): Names of the GCMs to use for simulated values.
    """

    # Confirm the supplied inputs are correct 
    check_epw_inputs(epw_filename,lon,lat,
                     epw_output_filename)

    # Confirm Accurate variable name
    check_epw_variable_name(epw_variable_name)

    # Confirm correct GCM files supplied given variable name
    
    if epw_filename != None:
        # Get the coordinates from the weather file
        epw_coords = get_epw_coordinates(epw_filename)
        lon = epw_coords[0]
        lat = epw_coords[1]

    # Get the present and future climate data.

    # Get the data from epw file and the headers from the epw.
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    # Morph columns of EPW dataframe based on selected options
    #----------------------------------------------------------
    # Dry Bulb Temperature
    if epw_variable_name == 'dry_bulb_temperature':
        #Separate Tasmax and Tasmin files from the inputs
        tx_ix = np.array([i for i,gcm in enumerate(present_climate_files) if 'tasmax' in gcm])
        tasmax_present_climate_files = np.array(present_climate_files)[tx_ix]
        tn_ix = np.array([i for i,gcm in enumerate(present_climate_files) if 'tasmin' in gcm])
        tasmin_present_climate_files = np.array(present_climate_files)[tn_ix]
        tx_ix = np.array([i for i,gcm in enumerate(future_climate_files) if 'tasmax' in gcm])
        tasmax_future_climate_files = np.array(future_climate_files)[tx_ix]
        tn_ix = np.array([i for i,gcm in enumerate(future_climate_files) if 'tasmin' in gcm])
        tasmin_future_climate_files = np.array(future_climate_files)[tn_ix]

        epw_dbt_morph = generate_dry_bulb_temperature(epw_data[epw_variable_name],
                                                      epw_data['datetime'],
                                                      lon,lat,
                                                      tasmax_present_climate_files,
                                                      tasmax_future_climate_files,
                                                      tasmin_present_climate_files,
                                                      tasmin_future_climate_files,
                                                      present_range,future_range,
                                                      factor)
        epw_data[epw_variable_name] = epw_dbt_morph
        write_epw_data(epw_data, headers, epw_output_filename)  
        return(epw_dbt_morph)
    #----------------------------------------------------------
    # Dewpoint Temperature
    if epw_variable_name == 'dew_point_temperature':
        print('Dewpoint')
        epw_dwpt_morph = generate_dewpoint_temperature(epw_data[epw_variable_name],
                                                       epw_data['datetime'],
                                                       lon,lat,
                                                       present_climate_files,
                                                       future_climate_files,
                                                       present_range,future_range,
                                                       factor)
        epw_data[epw_variable_name] = epw_dwpt_morph
        write_epw_data(epw_data, headers, epw_output_filename)  
        return(epw_dwpt_morph)
    #----------------------------------------------------------
    # Horizontal Radiation
    if epw_variable_name == 'global_horizontal_radiation':
        print('Both Horizontal Radiation Series')
        epw_hr_morph = generate_horizontal_radiation(epw_data['global_horizontal_radiation'],
                                                     epw_data['diffuse_horizontal_radiation'],
                                                     epw_data['datetime'],
                                                     lon,lat,
                                                     present_climate_files,
                                                     future_climate_files,
                                                     present_range,future_range,
                                                     factor)
        epw_data['global_horizontal_radiation'] = epw_hr_morph[:,0]
        epw_data['diffuse_horizontal_radiation'] = epw_hr_morph[:,1]
        write_epw_data(epw_data, headers, epw_output_filename)  
        return(epw_hr_morph)  
    #----------------------------------------------------------
    # Variables Morphed by stretch only
    stretch_variables = ['direct_normal_radiation','atmospheric_station_pressure',
                         'relative_humidity','windspeed',
                         'total_sky_cover','opaque_sky_cover','liquid_precipitation_quantity']
    if epw_variable_name in stretch_variables:
        cdf_vars = {'direct_normal_radiation': 'clt', 
                    'atmospheric_station_pressure': 'psl',
                    'relative_humidity': 'rhs',
                    'windspeed': 'wspd',
                    'total_sky_cover': 'clt',
                    'opaque_sky_cover': 'clt',
                    'liquid_precipitation_quantity': 'pr'}

        morphing_functions = {'direct_normal_radiation': morph_direct_normal_radiation, 
                              'atmospheric_station_pressure': morph_atmospheric_station_pressure,
                              'relative_humidity': morph_relative_humidity,
                              'windspeed': morph_windspeed,
                              'total_sky_cover': morph_total_sky_cover,
                              'opaque_sky_cover': morph_opaque_sky_cover,
                              'liquid_precipitation_quantity': morph_liquid_precip_quantity}

        cdfvariable = cdf_vars.get(epw_variable_name,'Missing EPW Variable')
        morphing_function = morphing_functions.get(epw_variable_name,'Missing EPW Variable')
        epw_var_morph = generate_stretched_series(epw_data[epw_variable_name],
                                                  epw_data['datetime'],
                                                  lon,lat,
                                                  cdfvariable,
                                                  present_climate_files,
                                                  future_climate_files,
                                                  present_range,future_range,
                                                  morphing_function,
                                                  factor)
        epw_data[epw_variable_name] = epw_var_morph
        write_epw_data(epw_data, headers, epw_output_filename)  
        return(epw_var_morph)  

    # Write the data out to the epw file.
    # write_epw_data(epw_rad_morph, headers, epw_output_filename)  
    return(epw_output_filename)


#-----------------------------------------------------------------
#-----------------------------------------------------------------

def get_epw_header(epw_file: IO) -> str:
    """get_epw_header(IO)

        Extracts the header from an epw file and returns it.

        Args:
            epw_file(IO): An open epw file

        Returns: (str): A string consisting of the header, usually the
            first 8 rows of the file
    """
    pos = epw_file.tell()  # Save the current position
    epw_file.seek(0)
    rv = ''.join([epw_file.readline() for _ in range(8)])
    epw_file.seek(pos)  # Reset the stream position
    return rv


def write_epw_data(data: list, headers: str, filename: str):
    """ write_epw_data(list, str, str)

        Combines the passed headers and data into an epw file with
        the name filename.

        Args:
            data(list): A list of lists, each inner list is a row of
                    data to be written to the epw file.
            header(str): The header string from the original epw file.
            filename(str): The name of the file to be written.
    """

    # Start with the headers and build from there
    epw_file = headers

    for ix in range(0, data.shape[0]):
        data_row = data.loc[ix].values.tolist()
        # epw files mandate that if the -3rd position is 999.000 that it is
        # missing. This is an issue because 999.000 is not a valid float, as
        # the trailing 0's are omitted. However, we can assume that if the
        # value is 999.0, that it is missing, and therefore we should add a
        # string of 999.000 rather than 999.0.
        if data_row[-3] == 999.0:
            data_row[-3] = "999.000"
        # The same logic as above applies, except with 0.0 and 0.0000.
        if data_row[-6] == 0:
            data_row[-6] = "0.0000"
        data_row[-8] = format(data_row[-8], '08d')

        # Get the date that this row is on, and assemble that into the first
        # 5 entries of the row.
        row_date = data_row[0]
        csv_row = [str(row_date.year), str(row_date.month),
                   str(row_date.day), str(row_date.hour+1),
                   str(row_date.minute)]

        # Afterwards, append strings of each cell to the csv_row list so that
        # we have a list of the exact strings that we want written into this
        # line of the epw file.
        for cell in data_row[1:]:
            csv_row.append(str(cell))

        # Finally, write that list to the epw_file string (and seperate each
        # entry in the list with a comma).
        epw_file += ",".join(csv_row) + "\n"

    # Write the generated string to the passed file.
    # We pre-generate the string as it is much quicker to append to a
    # string than it is to write to a file.
    with open(filename, "w+") as epw:
        epw.write(epw_file)

def get_epw_summary_values(epw_data: pandas.Series, dates: pandas.Series, 
                           factor: str, operator: str):
    """ get_monthly_values(list)

        Calculates monthly averages of daily max, min and average from the passed epw data, and returns
        a dataframe of those values.

        Args:
            epw_data(pandas.Series): The data read from the epw that we will be
                    averaging with.
            dates(pandas.Series): The datetime series from the pandas DataFrame
            factor(str): The date factor with which to calculate summary statistics
            operator(str): The summary statistic function.

        Returns:
            A dataframe of summary data.
    """
    #----------------------------------------------------
    # Prefer to use something like this with specified agg factor and operator
    input_dict = {'datetime': dates, 'data': epw_data}
    input_df = pandas.DataFrame(input_dict, columns=['datetime', 'data'])
    aggregate_values = input_df.groupby(input_df['datetime'].dt.strftime(factor)).agg(operator)
    # Could convert to string and back to dates using dataframe index, but taking max of time
    # takes only one line.
    aggregate_dates = input_df.groupby(input_df['datetime'].dt.strftime(factor)).max()['datetime']
    agg_dict = {'datetime':aggregate_dates,'data':aggregate_values['data']}
    return pandas.DataFrame(agg_dict, columns=['datetime', 'data'])
    
    #----------------------------------------------------

    # Old way
    #hourly_dict = {'datetime':dates,'hourly':epw_data}
    #hourly_df = pandas.DataFrame(hourly_dict,columns=['datetime','hourly'])    

    #daily_max = hourly_df.groupby(hourly_df['datetime'].dt.strftime('%m %d')).max()
    #monthly_max = daily_max.groupby(daily_max['datetime'].dt.strftime('%m')).mean()
    #daily_mean = hourly_df.groupby(hourly_df['datetime'].dt.strftime('%m %d')).mean()
    #daily_mean['datetime'] = daily_max['datetime'] ##FIXME this doesn't seem correct
    #monthly_mean = daily_mean.groupby(daily_mean['datetime'].dt.strftime('%m')).mean()
    #daily_min = hourly_df.groupby(hourly_df['datetime'].dt.strftime('%m %d')).min()
    #monthly_min = daily_min.groupby(daily_min['datetime'].dt.strftime('%m')).mean()

    #monthly_dict = {'Max':monthly_max['hourly'],'Min':monthly_min['hourly'],'Mean':monthly_mean['hourly']}
    #monthly_df = pandas.DataFrame(monthly_dict,columns=['Max','Min','Mean'])    
    #print(monthly_df)
    #return monthly_df


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# PRISM offset weather files


def get_epw_coordinates(filename):
    """ get_epw_coordinates(filename)
        Opens the epw file to obtain the first row and extracts the coordinates
        for the epw file
        Args:
            filename(str): An epw filename
    """
    nc = pandas.read_csv(filename, sep=',', header=None, nrows=1)
    # FIXME Replace with pattern matching
    rv = (float(nc[7].values), float(nc[6].values))
    return(rv)


def list_of_epw_coordinates(files):
    """ list_of_epw_coordinates(files)
        Obtains the spatial coordinates for all supplied epw files
        Args:
            files(list): A list of all available epw files
    """
    coords = np.zeros((len(files), 2))
    for i, file in enumerate(files):
        b = get_epw_coordinates(file)
        coords[i, :] = b
    return(coords)


def find_closest_epw_file(coords):
    """ find_closest_epw_file(coords)
        Loops through all epw files in the weather file directory, produces a list of 
        coordinates for all available files and finds the file nearest to the coords
        Args:
            coords(float,float): The longitude and latitude to compare with the weather files.
    """
    print(coords)
    # FIXME with non-hard coded location
    read_dir = "/storage/data/projects/rci/weather_files/wx_files/"
    files = glob.glob(read_dir+'*.epw')

    coord_data = list_of_epw_coordinates(files)
    wx_index = np.sum(np.square(coord_data-coords), axis=1).argmin()
    wx_selected = files[wx_index]
    return(wx_selected)


def prism_ncfile(varname):
    """ prism_ncfile(varname)
        Returns an opened netcdf object for a PRISM climatology file.
        Args:
            varname(str): Variable name
    """
    # FIXME with non-hard coded location
    fname = '/storage/data/projects/rci/weather_files/PRISM/' + varname + '_lm_subset.nc'
    dst = cdf.Dataset(fname, 'r')
    return dst


def get_prism_indices(nc, coords):
    """ get_prism_indices(nc, coords)
        Finds the nearest PRISM grid cell to the supplied coordinates and 
        returns the cell indices for the location.
        Args:
            nc (Open PRISM netcdf object)
            coords(float,float): Lon/Lat coordinates
    """
    lon_data = nc.variables["lon"][:]
    lon_index = np.absolute(lon_data - coords[0]).argmin()
    lat_data = nc.variables["lat"][:]
    lat_index = np.absolute(lat_data - coords[1]).argmin()
    rv = [lon_index, lat_index]
    return(rv)


def prism_read(nc, cells, varname):
    """ prism_read(nc, cells, varname)
        Returns the monthly PRISM climatologies from the PRISM netcdf file
        for the provided cell indices
        Args:
            nc (Open PRISM netcdf object)
            cells(int,int): PRISM cell indices
            varname(str): PRISM variable name                      
    """
    data = nc.variables[varname][:, cells[0], cells[1]]
    return(data[0:12, ])  # 13 entries, do not need the annual climatology (13)


def prism_tas(nc, cells):
    """ prism_tas(nc, cells, varname)
        Returns the monthly mean temperature PRISM climatologies from the PRISM netcdf file
        for the provided cell indices
        Args:
            nc (Open PRISM netcdf object)
            cells(int,int): PRISM cell indices
    """
    ncx = prism_ncfile('tmax')
    tmax = prism_read(ncx, cells, 'tmax')
    ncn = prism_ncfile('tmin')
    tmin = prism_read(ncn, cells, 'tmin')
    tas = np.divide(tmax + tmin, 2.0)
    return(tas)


def adjust_epw_with_prism(epw_data, prism_diff):
    """ adjust_epw_with_prism(epw_data,prism_diff)
        Adds the PRISM temperature offset to the EPW dry bulb temp and returns
        a new EPW dataframe
        Args:
            epw_data (pandas dataframe): Data from EPW file
            prism_diff(range): 12 monthly PRISM temperature offsets
    """
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    print('Months')
    new_epw = epw_data.copy()
    months = range(1, 13)
    # FIXME the SetWithCopyWarning for this assignment
    for mn in months:
        #new_epw.dry_bulb_temperature[epw_months == mn] = round(epw_data.dry_bulb_temperature[epw_months == mn] + prism_diff[mn-1],1)
        new_epw.ix[epw_months == mn, 'dry_bulb_temperature'] = round(
            new_epw.ix[epw_months == mn, 'dry_bulb_temperature'] + prism_diff[mn-1], 1)
    return(new_epw)


def gen_prism_offset_weather_file(lat: float,
                                  lon: float,
                                  ):
    """ gen_prism_offset_file(float, float)

        Generates an epw file based on a provided location by finding the nearest
        weather file to the supplied coordinates and applying an offset to the temperature
        series based on PRISM climatologies (1981-2010 for now).

        Args:
            lat(float): The latitude to read data from climate files.

            lon(float): The logitude to read data from the climate files.
    """

    coords = (lon, lat)
    # Search through all weather files for the closest to the coords
    epw_closest = find_closest_epw_file(coords)

    # Return the coordinates of the closest epw file
    epw_closest_coords = get_epw_coordinates(epw_closest)
    print(epw_closest_coords)
    # Any PRISM climatology file to grab coordinates
    nc = prism_ncfile('tmax')

    print('Closest PRISM cell to supplied coords')
    prism_cell = get_prism_indices(nc, coords)
    print(prism_cell)
    prism_loc_tas = prism_tas(nc, prism_cell)

    print('PRISM coords of cell closest to EPW File')
    epw_cell = get_prism_indices(nc, epw_closest_coords)
    print(epw_cell)
    prism_epw_tas = prism_tas(nc, epw_cell)

    prism_diff = prism_loc_tas - prism_epw_tas
    print(prism_diff)

    # Get the data from epw file and the headers from the epw.
    with open(epw_closest) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    epw_offset = adjust_epw_with_prism(epw_data, prism_diff)
    print(epw_offset.shape)
    # Write the data out to the epw file.
    epw_output_filename = "/storage/data/projects/rci/weather_files/wx_files/TEST.epw"
    write_epw_data(epw_offset, headers, epw_output_filename)
