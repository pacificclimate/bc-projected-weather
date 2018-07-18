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
# Conceptual Core Morphing Functions

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
    # FIX Possible fix
    # Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    # This version would allow specifying the time factor
    # Daily average temperatures
    ##epw_daily_averages = get_epw_summary_values(epw_data['dry_bulb_temperature'],
    ##                                          epw_data['datetime'],'%Y %m %d',mean)
    ### Monthly average temperatures
    ##epw_monthly_averages = get_epw_summary_values(epw_daily_averages['dry_bulb_temperature'],
    ##                                          epw_daily_averages['datetime'],'%m',mean)
   
    epw_averages = get_epw_summary_values(epw_tas,
                                          epw_dates,'%m','mean')
 
    # FIX to replace with alternative to if statement if possible
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
            (epw_averages['Max'][uf-1] -
             epw_averages['Min'][uf-1])
        anoms = epw_tas[ix] - epw_averages['Mean'][uf-1]
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
    # FIX - replace with specified time factor averaging (like DBT)
    epw_averages = get_epw_summary_values(epw_dwpt,
                                          epw_dates,factor,'mean')
 
    # FIX to replace with alternative to if statement if possible
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed_dwpt = np.zeros(len(epw_dwpt))

    for uf in unique_factors:
        ix = epw_factor == uf
        shift = epw_averages['Mean'][uf-1] + dewpoint_delta[uf-1]
        stretch = (epw_dwpt[ix] - epw_averages['Mean'][uf-1]) * dewpoint_alpha[uf-1]
        morphed_dwpt[ix] = round(shift + stretch,1)
    return(morphed_dbt)


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
    # FIX - replace with specified time factor averaging (like DBT)
    epw_averages = get_epw_summary_values(epw_dnr,
                                          epw_dates,factor,'mean')
 
    # FIX to replace with alternative to if statement if possible
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed = np.zeros(len(epw_dnr))

    for uf in unique_factors:
        ix = epw_factor == uf
        stretch = alpha[uf-1]
        morphed[ix] = epw_dnr / stretch
    return(round(morphed,0))

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
    # FIX - replace with specified time factor averaging (like DBT)
    epw_ghr_averages = get_epw_summary_values(epw_ghr,
                                          epw_dates,factor,'mean')
    epw_dhr_averages = get_epw_summary_values(epw_dhr,
                                          epw_dates,factor,'mean')
 
    # FIX to replace with alternative to if statement if possible
    if (factor == '%m'):
        epw_factor = pandas.DatetimeIndex(epw_dates).month
    if (factor == '%m %d'):
        epw_factor = pandas.DatetimeIndex(epw_dates).day
    unique_factors = epw_factor.unique()
    morphed_rad = np.zeros(len(epw_ghr),2)

    for uf in unique_factors:
        ix = epw_factor == uf
        global_hz_rad = epw_ghr[ix]
        global_hz_rad[global_hz_rad == 0] = 1
        diffuse_hz_rad = epw_dhr[ix]
        diffuse_to_total_ratio = diffuse_hz_rad / global_hz_rad
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
        morphed[ix] = epw_data * stretch
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
    return(round(morphed_rhs,0))

# ------------------------------------------------------
def morph_surface_pressure(epw_psl: pandas.Series,
                           epw_dates: pandas.Series,
                           psl_alpha: list,
                           factor: str
                            ):
    morphed_psl = morph_by_stretch(epw_psl,epw_dates,
                                   psl_alpha,factor)    
    return(round(morphed_rhs,0))
 
# ------------------------------------------------------
def morph_windspeed(epw_wspd: pandas.Series,
                    epw_dates: pandas.Series,
                    wspd_alpha: list,
                    factor: str
                   ):
    morphed_wspd = morph_by_stretch(epw_wspd,epw_dates,
                                   wspd_alpha,factor)    
    return(round(morphed_wspd,1))

# ------------------------------------------------------

def morph_total_sky_cover(epw_tsc: pandas.Series,
                          epw_dates: pandas.Series,
                          tsc_alpha: list,
                          factor: str
                         ):
    morphed_tsc = morph_by_stretch(epw_tsc,epw_dates,
                                   tsc_alpha,factor)    
    morphed_tsc[morphed_tsc > 10] = 10    
    return(round(morphed_tsc,0))

# ------------------------------------------------------

def morph_opaque_sky_cover(epw_osc: pandas.Series,
                           epw_dates: pandas.Series,
                           osc_alpha: list,
                           factor: str
                         ):
    morphed_osc = morph_by_stretch(epw_osc,epw_dates,
                                   osc_alpha,factor)    
    morphed_osc[morphed_osc > 10] = 10    
    return(round(morphed_osc,0))

# ------------------------------------------------------
# Unlikely to be modified due to data contraints
# def morph_wind_direction():
# def morph_snow_depth():
# def morph_liquid_precip_depth():
# def morph_precip_quantity():

# -----------------------------------

def generate_dry_bulb_temperature(epw_tas: pandas.Series,
                                  epw_dates: pandas.Series,
                                  lon: float, lat: float,
                                  tasmax_present_gcm_files: list,
                                  tasmax_future_gcm_files: list,
                                  tasmin_present_gcm_files: list,
                                  tasmin_future_gcm_files: list,
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
    print('Tasmax')
    tasmax_present = get_ensemble_averages(cdfvariable='tasmax',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmax_present_gcm_files,
                                           time_range=present_range,
                                           factor=factor)
    tasmax_future = get_ensemble_averages(cdfvariable='tasmax',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmax_future_gcm_files,
                                           time_range=future_range,
                                           factor=factor)
    tasmax_delta = tasmax_future['mean'] - tasmax_present['mean']
    
    print('Tasmin')
    tasmin_present = get_ensemble_averages(cdfvariable='tasmin',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmin_present_gcm_files,
                                           time_range=present_range,
                                           factor=factor)
    tasmin_future = get_ensemble_averages(cdfvariable='tasmin',
                                           lon=lon, lat=lat,
                                           gcm_files=tasmin_future_gcm_files,
                                           time_range=future_range,
                                           factor=factor)
    tasmin_delta = tasmin_future['mean'] - tasmin_present['mean']

    # One could also to use change in variability instead of change in diurnal cycle
    # (as is done in the "stretch" definition for other variables in Belcher).
    tas_delta = (tasmax_future['mean']+tasmin_future['mean'])/2 - (
        tasmax_present['mean']+tasmin_present['mean'])/2
    
    morphed_epw_tas = morph_dry_bulb_temperature(epw_tas,epw_dates,
                                                 np.mean(tasmax_delta,axis=1),
                                                 np.mean(tasmin_delta,axis=1),
                                                 np.mean(tas_delta,axis=1),
                                                 factor)
    print('Existing TAS')
    print(epw_tas[0:11])
    print('Morphed TAS')
    print(morphed_epw_tas[0:11])
    return morphed_epw_tas

def morph_dewpoint_temperature(epw_data: pandas.DataFrame,
                               lon: float, lat: float,
                               gcms: list,
                               present_range: range,
                               future_range: range
                               ):
    """ morph_dewpoint_temperature(pandas.DataFrame, float,float,str,range,range)

        This function takes in data from the epw file, and returns 
        the dataframe with the dewpoint temperature column having been
        replaced with future ("morphed") versions of dewpoint temperature.

        Args:
            data(DataFrame): A list of data representing the csv line found
                    in the epw file. IE one line of data, that has been
                    turned into a list (rather than a string read directly
                    from the file.)
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            gcm(str): The gcm name.
            present_range(range): Year bounds for the present climate.

            future_range(range): Year bounds for the future climate.

        Returns:
            a DataFrame with future dry bulb temperature data.
    """
    # Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    epw_monthly_averages = get_monthly_values(
        epw_data['dew_point_temperature'], epw_data['datetime'])
    # FIXME with non-hard coded location
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/"

    # Dewpoint
    print("Dewpoint")
    dewpoint_alpha_delta = get_ensemble_alpha_and_delta(cdfvariable='dewpoint', lon=lon, lat=lat,
                                                        gcms=gcms, gcm_dir=gcm_dir, ds_type="day",
                                                        present_suffix="_19500101-21001231.nc",
                                                        future_suffix="_19500101-21001231.nc",
                                                        present_range=present_range, future_range=future_range)

    dewpoint_std = get_ensemble_alpha_and_delta(cdfvariable='dewpoint', lon=lon, lat=lat,
                                                gcms=gcms, gcm_dir=gcm_dir, ds_type="day",
                                                present_suffix="_19500101-21001231.nc",
                                                future_suffix="_19500101-21001231.nc",
                                                present_range=present_range, future_range=future_range,
                                                std='sigma')
    print('Delta')
    print(dewpoint_alpha_delta['delta'])
    print('Alpha')
    print(dewpoint_std['alpha'])

    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1, 13)
    for mn in months:
        shift = epw_monthly_averages['Mean'][mn -
                                             1] + dewpoint_alpha_delta['delta'][mn-1]
        stretched_anoms = (epw_data.ix[epw_months == mn, 'dew_point_temperature'] -
                           epw_monthly_averages['Mean'][mn-1]) * dewpoint_std['alpha'][mn-1]
        morphed_dewpoint = shift + stretched_anoms
        new_epw.ix[epw_months == mn, 'dew_point_temperature'] = round(
            morphed_dewpoint, 1)
    return new_epw


def stretch_gcm_variable(epw_data: pandas.DataFrame,
                         lon: float, lat: float,
                         epw_variable: str,
                         netcdf_variable: str,
                         gcms: list,
                         present_range: range,
                         future_range: range
                         ):
    """ stretch_gcm_variable(pandas.DataFrame, float,float,str,range,range)

        This function takes in data from the epw file, and returns 
        the dataframe with the specified variable having been
        replaced with future ("stretched) versions. The GCM data is from
        the "building_code" directory.

        Args:
            data(DataFrame): A list of data representing the csv line found
                    in the epw file. IE one line of data, that has been
                    turned into a list (rather than a string read directly
                    from the file.)
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            epw_variable(str): The name of the column to stretch
            netcdf_variable(str): The name of the variable to read from the 
                                  climate file            
            gcm(str): The gcm name.
            present_range(range): Year bounds for the present climate.
            future_range(range): Year bounds for the future climate.

        Returns:
            a DataFrame with one stretched column of data.
    """
    # Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    # FIXME with non-hard coded location
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/"

    # Find the ensemble average value for alpha
    alpha_delta = get_ensemble_alpha_and_delta(cdfvariable=netcdf_variable, lon=lon, lat=lat,
                                               gcms=gcms, gcm_dir=gcm_dir, ds_type="day",
                                               present_suffix="_19500101-21001231.nc",
                                               future_suffix="_19500101-21001231.nc",
                                               present_range=present_range, future_range=future_range)
    alpha_ens = alpha_delta['alpha']
    print('Alpha Ensemble')
    print(alpha_ens)
    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1, 13)
    for mn in months:
        morphed_data = epw_data.ix[epw_months ==
                                   mn, epw_variable] * alpha_ens[mn-1]
        if epw_variable == "wind_speed":
            morphed_data = round(morphed_data, 1)
        else:
            morphed_data = round(morphed_data, 0).astype(int)
        if epw_variable == "relative_humidity":
            morphed_data[morphed_data > 100] = 100
        if epw_variable == "total_sky_cover":
            morphed_data[morphed_data > 10] = 10
        if epw_variable == "opaque_sky_cover":
            morphed_data[morphed_data > 10] = 10
        new_epw.ix[epw_months == mn, epw_variable] = morphed_data
    return new_epw


def morph_radiation(epw_data: pandas.DataFrame,
                    lon: float, lat: float,
                    gcms: list,
                    present_range: range,
                    future_range: range
                    ):
    """ morph_radiation(pandas.DataFrame, float,float,list,range,range)

        This function takes in data from the epw file, and returns 
        the dataframe with the global horizontal, direct normal, diffuse horizontal radiation and sky
        cover variables having been replaced with future versions. The GCM data is from
        the "building_code" directory.

        Args:
            data(DataFrame): A list of data representing the csv line found
                    in the epw file. IE one line of data, that has been
                    turned into a list (rather than a string read directly
                    from the file.)
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            gcms(list): The ensemble of gcm names.
            present_range(range): Year bounds for the present climate.
            future_range(range): Year bounds for the future climate.

        Returns:
            a DataFrame with one stretched column of data.
    """
    # Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    # FIXME with non-hard coded location
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/"

    # Total Horizontal Solar Radiation
    print("RSDS")
    rsds_alpha_delta = get_ensemble_alpha_and_delta(cdfvariable='rsds', lon=lon, lat=lat,
                                                    gcms=gcms, gcm_dir=gcm_dir, ds_type="day",
                                                    present_suffix="_19500101-21001231.nc",
                                                    future_suffix="_19500101-21001231.nc",
                                                    present_range=present_range, future_range=future_range)
    alpha_rsds = rsds_alpha_delta['alpha']
    print(alpha_rsds)
    # Cloud Cover
    print("CLT")
    clt_alpha_delta = get_ensemble_alpha_and_delta(cdfvariable='clt', lon=lon, lat=lat,
                                                   gcms=gcms, gcm_dir=gcm_dir, ds_type="day",
                                                   present_suffix="_19500101-21001231.nc",
                                                   future_suffix="_19500101-21001231.nc",
                                                   present_range=present_range, future_range=future_range)
    alpha_clt = clt_alpha_delta['alpha']
    print(alpha_clt)
    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1, 13)

    for mn in months:
        global_horiz_rad = epw_data.ix[epw_months ==
                                       mn, "global_horizontal_radiation"]
        diffuse_rad = epw_data.ix[epw_months ==
                                  mn, "diffuse_horizontal_radiation"]
        global_horiz_rad[global_horiz_rad == 0] = 1
        diffuse_to_total_ratio = diffuse_rad / global_horiz_rad
        morphed_global_horiz_rad = epw_data.ix[epw_months ==
                                               mn, "global_horizontal_radiation"] * alpha_rsds[mn-1]
        morphed_diffuse_horiz_rad = morphed_global_horiz_rad * diffuse_to_total_ratio

        new_epw.ix[epw_months == mn, "global_horizontal_radiation"] = round(
            morphed_global_horiz_rad, 0)
        new_epw.ix[epw_months == mn, "diffuse_horizontal_radiation"] = round(
            morphed_diffuse_horiz_rad, 0)

        normal_rad = epw_data.ix[epw_months == mn, "direct_normal_radiation"]
        clouds = epw_data.ix[epw_months == mn, "total_sky_cover"]
        opaque = epw_data.ix[epw_months == mn, "opaque_sky_cover"]
        morphed_clouds = clouds * alpha_clt[mn-1]
        morphed_clouds[clouds > 10] = 10
        morphed_opaque = opaque * alpha_clt[mn-1]
        morphed_opaque[opaque > 10] = 10

        # Assume direct normal radiation responds inversely proportionally to cloud cover changes
        morphed_normal_rad = normal_rad / alpha_clt[mn-1]

        new_epw.ix[epw_months == mn, "total_sky_cover"] = round(
            morphed_clouds, 0).astype(int)
        new_epw.ix[epw_months == mn, "opaque_sky_cover"] = round(
            morphed_opaque, 0).astype(int)
        new_epw.ix[epw_months == mn, "direct_normal_radiation"] = round(
            morphed_normal_rad, 0)

    return new_epw


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
    #input_dict = {'datetime': dates, 'data': epw_data}
    #input_df = pandas.DataFrame(input_dict, columns=['datetime', 'data'])
    #aggregate_values = input_df.groupby(input_df['datetime'].dt.strftime(factor)).agg(operator)
    #return aggregate_values
    # FIXME - How do you return daily dates if the operator is 'mean' instead of 'max' or 'min'
    #----------------------------------------------------

    # Existing way
    hourly_dict = {'datetime':dates,'hourly':epw_data}
    hourly_df = pandas.DataFrame(hourly_dict,columns=['datetime','hourly'])    

    daily_max = hourly_df.groupby(hourly_df['datetime'].dt.strftime('%m %d')).max()
    monthly_max = daily_max.groupby(daily_max['datetime'].dt.strftime('%m')).mean()
    daily_mean = hourly_df.groupby(hourly_df['datetime'].dt.strftime('%m %d')).mean()
    daily_mean['datetime'] = daily_max['datetime'] ##FIXME this doesn't seem correct
    monthly_mean = daily_mean.groupby(daily_mean['datetime'].dt.strftime('%m')).mean()
    daily_min = hourly_df.groupby(hourly_df['datetime'].dt.strftime('%m %d')).min()
    monthly_min = daily_min.groupby(daily_min['datetime'].dt.strftime('%m')).mean()

    monthly_dict = {'Max':monthly_max['hourly'],'Min':monthly_min['hourly'],'Mean':monthly_mean['hourly']}
    monthly_df = pandas.DataFrame(monthly_dict,columns=['Max','Min','Mean'])    
    return monthly_df

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
    print('GCM Ensemble')
    print(cdfvariable)
    # Leave options for monthly or daily averaging
    # FIX: prefer not to hard code the aggregated time length
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
        # Compute the ensemble averages here
        # ens_climatologies =  {'mean':np.mean(mean_aggregate,axis=1),
        #                       'std':np.mean(std_aggregate,axis=1)}
    ens_climatologies =  {'mean':mean_aggregate,
                          'std':std_aggregate}
    return ens_climatologies

def cftime_to_datetime(data_dates,calendar):
    # Convert cftime dates to string 
    ex_years = [date.strftime('%Y') for date in data_dates]
    ex_months = [date.strftime('%m') for date in data_dates]
    ex_days = [date.strftime('%d') for date in data_dates]
    # Concatenate to full date
    ymd = [x+'-'+y+'-'+z for x,y,z in zip(ex_years,ex_months,ex_days)]
    # FIX? Set irregular February dates to Feb 28
    if (calendar=='360_day'):
        ymd = [re.sub('-02-29','-02-28',x) for x in ymd]
        ymd = [re.sub('-02-30','-02-28',x) for x in ymd]
    # Convert to datetime object
    dates_list = [datetime.strptime(date, '%Y-%m-%d') for date in ymd]
    # Convert to array
    dates_array = np.asarray(dates_list)
    return(dates_array)

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
    print('Dates')
    data_dates = cdf.num2date(nc["time"][:], nc["time"].units,nc["time"].calendar)

    dates_array = cftime_to_datetime(data_dates,nc['time'].calendar)
    startyear,endyear = time_range

    t0 = np.argwhere(dates_array >= datetime(startyear, 1, 1)).min()
    print(dates_array[t0])
    print(data_dates[t0])
    if nc['time'].calendar == "360_day":
        tn = np.argwhere(dates_array <= datetime(endyear, 12, 30)).max()
    else:
        tn = np.argwhere(dates_array <= datetime(endyear, 12, 31)).max()
    print(dates_array[tn])
    print(data_dates[tn])

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

    data_dict = {'Time': dates_array[t0:tn+1], 'Data': data} ##{'Time': data_dates[t0:tn+1], 'Data': data}
    data_frame = pandas.DataFrame(data_dict, columns=['Time', 'Data'])
    time_mean = data_frame.groupby(
        data_frame['Time'].dt.strftime(factor)).mean().values
    time_std = data_frame.groupby(
        data_frame['Time'].dt.strftime(factor)).std().values
    data_clim = {'mean':time_mean, 'std':time_std}
    return data_clim


def gen_future_weather_file(epw_filename: str,
                            epw_output_filename: str,
                            present_range: range,
                            future_range: range,
                            gcms: list
                            ):
    """ gen_future_weather_file(float, float, range, range, str)

        Regenerates the passed epw file into a weather file represeting future
        data.

        Args:
            epw_filename(str): The path to the epw file to regenerate.

            epw_output_filename(str): The path to the future epw file to create

            present_range(range): The range of years that makes up "present"
                          for this particular run.

            future_range(range): The range of years that makes up "future" for
                         this particular run.
            gcm(list): Names of the GCMs to use for simulated values.
    """

    # Get the coordinates from the weather file
    epw_coords = get_epw_coordinates(epw_filename)
    lon = epw_coords[0]
    lat = epw_coords[1]
    print("Longitude: ")
    print(lon)
    print("Latitude: ")
    print(lat)
    # Get the present and future climate data.

    # Get the data from epw file and the headers from the epw.
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    # Morph columns of EPW dataframe based on selected options
    epw_dbt_morph = morph_dry_bulb_temperature(
        epw_data, lon, lat, gcms, present_range, future_range)

    print('Dewpoint')
    epw_dpt_morph = morph_dewpoint_temperature(
        epw_dbt_morph, lon, lat, gcms, present_range, future_range)
    print('Relative Humidity')
    epw_rhs_morph = stretch_gcm_variable(
        epw_dpt_morph, lon, lat, "relative_humidity", "rhs", gcms, present_range, future_range)
    print('Air Pressure')
    epw_psl_morph = stretch_gcm_variable(
        epw_rhs_morph, lon, lat, "atmospheric_station_pressure", "psl", gcms, present_range, future_range)
    print('Windspeed')
    epw_wspd_morph = stretch_gcm_variable(
        epw_psl_morph, lon, lat, "wind_speed", "wspd", gcms, present_range, future_range)
    print('Radiation and Sky Cover')
    epw_rad_morph = morph_radiation(
        epw_wspd_morph, lon, lat, gcms, present_range, future_range)

    # Write the data out to the epw file.
    write_epw_data(epw_rad_morph, headers, epw_output_filename)
    return(epw_output_filename)
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
