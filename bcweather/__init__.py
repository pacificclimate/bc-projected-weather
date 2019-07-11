from datetime import datetime
import re

import numpy as np
from netCDF4 import Dataset
import netCDF4 as cdf
from typing import IO
import pandas
import glob
import os as os

from .epw import epw_to_data_frame

# -----------------------------------
# Core Morphing Functions
# Take the daily/monthly delta factors, compute the morphed series
# and return the new series


def morph_dry_bulb_temperature(epw_tas: pandas.Series,
                               epw_dates: pandas.Series,
                               alpha_tas: list,
                               delta_tas: list,
                               factor: str,
                               rlen: int
                               ):
    """ morph_dry_bulb_temperature(pandas.Series,pandas.Series,
                                   list,list,str,int)

        This function takes in hourly temperature data from the epw file,
        the delta factors for max, min and mean temperature, and returns
        the "morphed" future dry bulb temperature.

        Args:
            epw_tas(Series): A hourly temperature column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            alpha_tas(list): Lists of change factors for
                            max,min temperature.
            delta_tas(list): Lists of change factors for
                            mean temperature.
            factor(str): Date factor with which to average.
            rlen(int) : Rolling average window (1 if not needed).
        Returns:
            an array with future dry bulb temperature data.
    """

    # ----------------------------------------------------------
    # Time averaged temperatures based on specific factor
    if (factor == 'monthly'):
        fac = '%m'
    if (factor == 'daily') or (factor == 'roll'):
        fac = '%m %d'

    epw_tas_dict = {'datetime': epw_dates, 'data': epw_tas}
    epw_tas_df = pandas.DataFrame(epw_tas_dict,
                                  columns=['datetime', 'data'])

    epw_daily_averages = get_epw_summary_values(epw_tas_df,
                                                '%Y %m %d', 'mean', 1)
    epw_tas_averages = get_epw_summary_values(epw_daily_averages,
                                              fac, 'mean', rlen)['data']
    epw_daily_max = get_epw_summary_values(epw_tas_df,
                                           '%Y %m %d', 'max', 1)
    epw_tas_max = get_epw_summary_values(epw_daily_max,
                                         fac, 'mean', rlen)['data']
    epw_daily_min = get_epw_summary_values(epw_tas_df,
                                           '%Y %m %d', 'min', 1)
    epw_tas_min = get_epw_summary_values(epw_daily_min,
                                         fac, 'mean', rlen)['data']

    epw_factor = pandas.DatetimeIndex(epw_dates).dayofyear
    unique_factors = epw_factor.unique()
    morphed_dbt = np.zeros(len(epw_tas))

    for uf in unique_factors:
        ix = epw_factor == uf
        shift = epw_tas[ix] + delta_tas[uf-1]
        alpha = alpha_tas[uf-1] / \
            (epw_tas_max[uf-1] - epw_tas_min[uf-1])
        anoms = epw_tas[ix] - epw_tas_averages[uf-1]
        morphed_dbt[ix] = round(shift + alpha * anoms, 1)
    return(morphed_dbt)

# ------------------------------------------------------


def morph_dewpoint_temperature(epw_dwpt: pandas.Series,
                               epw_dates: pandas.Series,
                               dewpoint_alpha: list,
                               dewpoint_delta: list,
                               factor: str,
                               rlen: int
                               ):
    """ morph_dewpoint_temperature(pandas.Series,pandas.Series,
                                   list,list,
                                   str)

        This function takes in hourly dewpoint temperature data
        from the epw file, the delta and alpha factors for
        dewpoint temperature, and returns the "morphed" future
        dewpoint temperature.

        Args:
            epw_dwpt(Series): A hourly dewpoint temperature column
                              from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            dewpoint_delta(list): Lists of shift factors for
                                  dewpoint temperature.
            dewpoint_alpha(list): Lists of stretch factors for
                                  dewpoint temperature.
            factor(str): Date factor with which to average.
            rlen(int) : Rolling average window (1 if not needed).

        Returns:
            an array with future dry bulb temperature data.
    """
    # Time averaged temperatures based on specific factor
    if (factor == 'monthly'):
        fac = '%m'
    if (factor == 'daily') or (factor == 'roll'):
        fac = '%m %d'

    epw_dwpt_dict = {'datetime': epw_dates, 'data': epw_dwpt}
    epw_dwpt_df = pandas.DataFrame(epw_dwpt_dict,
                                   columns=['datetime', 'data'])

    epw_daily_averages = get_epw_summary_values(epw_dwpt_df,
                                                '%Y %m %d',
                                                'mean', 1)
    epw_averages = get_epw_summary_values(epw_daily_averages,
                                          fac, 'mean', rlen)['data']
    epw_factor = pandas.DatetimeIndex(epw_dates).dayofyear
    unique_factors = epw_factor.unique()
    morphed_dwpt = np.zeros(len(epw_dwpt))

    for uf in unique_factors:
        ix = epw_factor == uf
        shift = epw_averages[uf-1] + dewpoint_delta[uf-1]
        stretch = (epw_dwpt[ix] - epw_averages[uf-1]) * dewpoint_alpha[uf-1]
        morphed_dwpt[ix] = round(shift + stretch, 1)
    return(morphed_dwpt)

# ------------------------------------------------------


def morph_direct_normal_radiation(epw_dnr: pandas.Series,
                                  epw_dates: pandas.Series,
                                  alpha: list
                                  ):
    """ morph_direct_normal_radiation(pandas.Series,pandas.Series,
                                      list,str,rlen)

        This function takes in hourly direct normal radiation data,
        the alpha factors for stretching, and returns the "morphed"
        future data. This is uses cloud cover to stretch (inversely
        proportional to cloud cover changes).

        Args:
            epw_dnr(Series): An hourly data column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            alpha(list): List of cloud cover stretch factors.
            factor(str): Date factor with which to average.
            rlen(int) : Rolling average window (1 if not needed).
        Returns:
            an array with future hourly epw data.
    """

    epw_factor = pandas.DatetimeIndex(epw_dates).dayofyear
    unique_factors = epw_factor.unique()
    morphed = np.zeros(len(epw_dnr))

    for uf in unique_factors:
        ix = epw_factor == uf
        stretch = alpha[uf-1]
        morphed[ix] = epw_dnr[ix] / stretch
    return(np.round(morphed.astype(int), 0))

# ------------------------------------------------------


def morph_horizontal_radiation(epw_ghr: pandas.Series, epw_dhr: pandas.Series,
                               epw_dates: pandas.Series,
                               alpha: list
                               ):
    """ morph_horizontal_radiation(pandas.Series,pandas.Series,
                                   pandas.Series,list,str)

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
            rlen(int) : Rolling average window (1 if not needed).
        Returns:
            an 2D array with future hourly epw data.
    """

    epw_factor = pandas.DatetimeIndex(epw_dates).dayofyear
    unique_factors = epw_factor.unique()
    morphed_rad = np.zeros((len(epw_ghr), 2))
    for uf in unique_factors:
        ix = epw_factor == uf
        global_hz_rad = epw_ghr[ix]
        flag = global_hz_rad == 0
        diffuse_hz_rad = epw_dhr[ix]
        diffuse_to_total_ratio = diffuse_hz_rad / global_hz_rad
        diffuse_to_total_ratio[flag] = 0
        morphed_global_hz_rad = global_hz_rad * alpha[uf-1]
        morphed_diffuse_hz_rad = morphed_global_hz_rad * diffuse_to_total_ratio
        morphed_rad[ix, 0] = round(morphed_global_hz_rad, 0)
        morphed_rad[ix, 1] = round(morphed_diffuse_hz_rad, 0)
    return(np.round(morphed_rad.astype(int), 0))

# ------------------------------------------------------


def morph_by_stretch(epw_data: pandas.Series,
                     epw_dates: pandas.Series,
                     alpha: list
                     ):
    """ morph_by_stretch(pandas.Series,pandas.Series,
                         list)

        This function takes in hourly weather file data,
        the alpha factors for stretching, and returns the "morphed"
        future data. This ise used by multiple epw variables.

        Args:
            epw_data(Series): An hourly data column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            alpha(list): List of stretch factors.
        Returns:
            an array with future hourly epw data.
    """
    epw_factor = pandas.DatetimeIndex(epw_dates).dayofyear
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
                            rhs_alpha: list
                            ):
    morphed_rhs = morph_by_stretch(epw_rhs, epw_dates,
                                   rhs_alpha)
    morphed_rhs[morphed_rhs > 100] = 100
    rv = np.asarray(morphed_rhs).astype(int)
    return(np.round(rv, 0))

# ------------------------------------------------------


def morph_atmospheric_station_pressure(epw_psl: pandas.Series,
                                       epw_dates: pandas.Series,
                                       psl_alpha: list
                                       ):
    morphed_psl = morph_by_stretch(epw_psl, epw_dates,
                                   psl_alpha)
    rv = np.asarray(morphed_psl).astype(int)
    return(np.round(rv, 0))

# ------------------------------------------------------


def morph_wind_speed(epw_wspd: pandas.Series,
                     epw_dates: pandas.Series,
                     wspd_alpha: list
                     ):
    morphed_wspd = morph_by_stretch(epw_wspd, epw_dates,
                                    wspd_alpha)
    rv = np.asarray(morphed_wspd).astype(float)
    return(np.round(rv, 1))

# ------------------------------------------------------


def morph_total_sky_cover(epw_tsc: pandas.Series,
                          epw_dates: pandas.Series,
                          tsc_alpha: list
                          ):
    morphed_tsc = morph_by_stretch(epw_tsc, epw_dates,
                                   tsc_alpha)
    morphed_tsc[morphed_tsc > 10] = 10
    rv = np.asarray(morphed_tsc).astype(int)
    return(np.round(rv, 0))

# ------------------------------------------------------


def morph_opaque_sky_cover(epw_osc: pandas.Series,
                           epw_dates: pandas.Series,
                           osc_alpha: list
                           ):
    morphed_osc = morph_by_stretch(epw_osc, epw_dates,
                                   osc_alpha)
    morphed_osc[morphed_osc > 10] = 10
    rv = np.asarray(morphed_osc).astype(int)
    return(np.round(rv, 0))

# ------------------------------------------------------


def morph_liquid_precip_quantity(epw_pr: pandas.Series,
                                 epw_dates: pandas.Series,
                                 pr_alpha: list
                                 ):
    morphed_pr = morph_by_stretch(epw_pr, epw_dates,
                                  pr_alpha)
    rv = np.asarray(morphed_pr).astype(int)
    return(np.round(rv, 0))

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
                                  alpha_tas_files: list,
                                  delta_tas_files: list,
                                  factor: str,
                                  rlen: int
                                  ):
    """ generate_dry_bulb_temperature(pandas.Series,pandas.Series,
                                      float,float,
                                      list,list,
                                      str,int)

        This function takes in data from the epw file, and returns
        the dataframe with the dry bulb temperature column having been
        replaced with future ("morphed") versions of dry bulb temperature.

        Args:
            epw_tas(Series): A hourly temperature column from the epw file.
            epw_dates(Series): The hourly dates from the epw file.
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            alpha_tas_files(list): The list of gcm files for alpha
            morphing factors
            delta_tas_files(list): The list of gcm files for delta
            morphing factors
            factor(str): Date factor with which to average.
            rlen(int): Rolling average window length.

        Returns:
            a list with future dry bulb temperature data.
    """
    # Morphing factors from the input gcm files
    alpha_tas = get_ensemble_averages(
        cdfvariable='alpha_tas',
        lon=lon, lat=lat,
        gcm_files=alpha_tas_files,
        factor=factor,
        rlen=rlen,
    )
    delta_tas = get_ensemble_averages(
        cdfvariable='tas',
        lon=lon, lat=lat,
        gcm_files=delta_tas_files,
        factor=factor,
        rlen=rlen
    )

    morphed_epw_tas = morph_dry_bulb_temperature(epw_tas, epw_dates,
                                                 alpha_tas['mean'],
                                                 delta_tas['mean'],
                                                 factor, rlen)
    return(morphed_epw_tas)

# -----------------------------------------------------------------


def generate_dewpoint_temperature(epw_dwpt: pandas.Series,
                                  epw_dates: pandas.Series,
                                  lon: float, lat: float,
                                  alpha_dwpt_files: list,
                                  delta_dwpt_files: list,
                                  factor: str,
                                  rlen: int
                                  ):
    """ generate_dewpoint_temperature(pandas.Series, pandas.Series,
                                      float,float,
                                      list,list,
                                      str,int)

        This function takes in dewpoint data from the epw file, and returns
        the future ("morphed") versions of dewpoint temperature.

        Args:
            data(Series): Dewpoint series from EPW file
            dates(Series): Dates from EPW file
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            alpha_dwpt_files(list):
            delta_dwpt_files(list):
            factor(str): Averaging factor
            ren(int): Window for rolling average.
        Returns:
            a numpy array of future dewpoint temperature data.
    """
    # Morphing factors from the input gcm files
    alpha_dwpt = get_ensemble_averages(
        cdfvariable='alpha_dewpoint',
        lon=lon, lat=lat,
        gcm_files=alpha_dwpt_files,
        factor=factor,
        rlen=rlen
    )
    delta_dwpt = get_ensemble_averages(
        cdfvariable='dewpoint',
        lon=lon, lat=lat,
        gcm_files=delta_dwpt_files,
        factor=factor,
        rlen=rlen
    )
    morphed_dwpt = morph_dewpoint_temperature(epw_dwpt, epw_dates,
                                              alpha_dwpt['mean'],
                                              delta_dwpt['mean'],
                                              factor, rlen)
    return(morphed_dwpt)

# -----------------------------------------------------------------


def generate_horizontal_radiation(epw_ghr: pandas.Series,
                                  epw_dhr: pandas.Series,
                                  epw_dates: pandas.Series,
                                  lon: float, lat: float,
                                  alpha_rsds_files: list,
                                  factor: str, rlen: int
                                  ):
    """ generate_horizontal_radiation(pandas.Series, pandas.Series,pandas.Series
                                      float,float,
                                      list,str,int)

        This function takes in global and diffuse horizontal radiation data
        from the epw file and returns the morphed versions.

        Args:
            data(Series): Global Horizontal Radiation series from EPW file
            data(Series): Diffuse Horizontal Radiation series from EPW file
            dates(Series): Dates from EPW file
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            alpha_rsds_files(list): Uses RSDS
            factor(str): Averaging factor
            rlen(int): Window for rolling average
        Returns:
            a 2D numpy array of future horizontal radiation data.
    """
    # Morphing factors from the input gcm files
    alpha_rsds = get_ensemble_averages(
        cdfvariable='rsds',
        lon=lon, lat=lat,
        gcm_files=alpha_rsds_files,
        factor=factor,
        rlen=rlen
    )
    morphed_horiz_rad = morph_horizontal_radiation(
        epw_ghr, epw_dhr, epw_dates,
        alpha_rsds['mean'])
    return(morphed_horiz_rad)

# ----------------------------------------------------------------


def generate_stretched_series(epw_data: pandas.Series,
                              epw_dates: pandas.Series,
                              lon: float, lat: float,
                              cdfvariable: str,
                              alpha_files: list,
                              morphing_function,
                              factor: str,
                              rlen: int
                              ):
    """ generate_stretched_series(pandas.Series, pandas.Series,
                               float,float,str,
                               list,str,int)

        This function takes in a single series of data from the epw file,
        and returns the morphed version.

        Args:
            data(Series): EPW Series
            dates(Series): Dates from EPW file
            lon(float): The longitude to read data from climate files
            lat(float): The latitude to read data from climate files
            cdfvariable(str): Netcdf variable
            alpha_files(list):
            morphing_function: Function to produce the stretched series
            factor(str): Averaging factor
            rlen(int) : Window for rolling average.
        Returns:
            a numpy array of future data.
    """

    alphas = get_ensemble_averages(cdfvariable=cdfvariable,
                                   lon=lon, lat=lat,
                                   gcm_files=alpha_files,
                                   factor=factor, rlen=rlen)

    morphed_epw = morphing_function(epw_data, epw_dates,
                                    alphas['mean'])

    return(morphed_epw)

# ----------------------------------------------------------------


def get_ensemble_averages(cdfvariable: str,
                          lon: float,
                          lat: float,
                          gcm_files: list,
                          factor: str,
                          rlen=int,
                          ):
    """ get_ensemble_averages(cdfvariable,lat,lon,gcms_files,time_range,
                              factor,rlen)

        Returns the climatological averages of the specified netcdf files.
        Args:
            cdfvariable(str): The variable to read from the netcdf file.
            lon(float): The longitude to read data from.
            lat(float): The latitude to read data from.
            gcm_files(list): Ensemble of GCMs to use
            time_range(list): The start and end years to read data from, to.
            factor(str): The time interval over which to average.
            rlen(int): The window for the rolling average if factor is 'roll'
        Returns:
            a dict with two numpy arrays.
            One for each time interval of the year.
    """
    # All averaging yields 365 values
    tlen = 365
    # Assemble the morphing factors
    mean_aggregate = np.zeros((tlen, len(gcm_files)))
    std_aggregate = np.zeros((tlen, len(gcm_files)))
    for i, gcm_file in enumerate(gcm_files):
        with Dataset(gcm_file) as f:
            file_climate = get_climate_data(
                f, lat, lon, cdfvariable, factor)
        mean_aggregate[:, i] = file_climate['mean'].flatten()
        std_aggregate[:, i] = file_climate['std'].flatten()

    mean_ens = np.nanmean(mean_aggregate, axis=1)
    std_ens = np.nanmean(std_aggregate, axis=1)
    if factor == 'roll':
        mean_ens_pd = pandas.Series(mean_ens)
        mean_ens_roll = mean_ens_pd.rolling(rlen,
                                            min_periods=1,
                                            center=True).mean()
        std_ens_pd = pandas.Series(std_ens)
        std_ens_roll = std_ens_pd.rolling(rlen,
                                          min_periods=1,
                                          center=True).mean()
        ens_clim = {'mean':
                    np.asarray(mean_ens_roll),
                    'std':
                    np.asarray(std_ens_roll)}
    else:
        ens_clim = {'mean':
                    mean_ens.flatten(),
                    'std':
                    std_ens.flatten()}
    return(ens_clim)

# ----------------------------------------------------------------


def cftime_to_datetime(data_dates, calendar):
    """ cftime_to_datetime(cftime, str)

        Converts the netcdf time series into a
        datetime object

        Args:
            data_dates(cftime): An open netCDF4.Dataset time series.
            calendar(str): The time calendar for conversion.
        Returns:
            a datetime series as an numpy array.
    """
    # Convert cftime dates to string
    ex_years = [date.strftime('%Y') for date in data_dates]
    ex_months = [date.strftime('%m') for date in data_dates]
    ex_days = [date.strftime('%d') for date in data_dates]
    # Concatenate to full date
    ymd = [x+'-'+y+'-'+z for x, y, z in zip(ex_years, ex_months, ex_days)]
    # Set irregular February dates to Feb 28 temporarily
    if (calendar == '360_day'):
        ymd = [re.sub('-02-29', '-02-28', x) for x in ymd]
        ymd = [re.sub('-02-30', '-02-28', x) for x in ymd]
    # Convert to datetime object
    dates_list = [datetime.strptime(date, '%Y-%m-%d') for date in ymd]
    # Convert to array
    dates_array = np.asarray(dates_list)
    return(dates_array)

# ----------------------------------------------------------------


def get_climate_data(nc: Dataset,
                     lat: float,
                     lon: float,
                     cdfvariable: str,
                     factor: str,
                     ):
    """ get_climate_data(Dataset, float, float, str, str)

        Gets a list of data for each day of each year within time_range from
        the climate file where the location is closest to lat, lon.

        Args:
            nc(Dataset): An open netCDF4.Dataset object.
            lat(float): The latitude to read data from.
            lon(float): The longitude to read data from.
            cdfvariable(str): The variable to read from the netcdf file.
            factor(str): The time factor with which to average.
        Returns:
            a dict with two lists.
            The first list contains the mean values, while the second
            list contains the standard deviations.
    """
    # Get a list of the dates in the climate file.
    data_dates = cdf.num2date(
        nc["time"][:], nc["time"].units, nc["time"].calendar)
    dates_array = cftime_to_datetime(data_dates, nc['time'].calendar)

    # Get the latitude of each location in the file.
    lat_data = nc.variables["lat"][:]
    # Get the logitude of each location in the file.
    lon_data = nc.variables["lon"][:]

    # Find the indices of the data with the closest lat and lon to
    # those passed.
    lat_index = np.absolute(lat - lat_data).argmin()
    lon_index = np.absolute(lon - lon_data).argmin()

    # Grab the actual relevant data from the file
    data = nc.variables[cdfvariable][:, lat_index, lon_index]

    dates_matrix = {'Time': dates_array, 'Data': data}
    data_frame = pandas.DataFrame(dates_matrix, columns=['Time', 'Data'])

    if factor == 'monthly':
        fac = '%m'
        reps = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        mon_mean = data_frame.groupby(
            data_frame['Time'].dt.strftime(fac)).mean().values
        mon_std = data_frame.groupby(
            data_frame['Time'].dt.strftime(fac)).std().values
        time_mean = np.repeat(mon_mean.flatten(), reps, axis=0)
        time_std = np.repeat(mon_std.flatten(), reps, axis=0)
    if (factor == 'daily') or (factor == 'roll'):
        fac = '%m %d'
        time_mean = data_frame.groupby(
            data_frame['Time'].dt.strftime(fac)).mean().values
        time_std = data_frame.groupby(
            data_frame['Time'].dt.strftime(fac)).std().values

    data_clim = {'mean': time_mean, 'std': time_std, 'time': dates_array}
    return(data_clim)

# ----------------------------------------------------------------


def check_epw_variable_name(epw_variable_name: str):
    from .epw import field_names
    if epw_variable_name not in field_names:
        print(epw_variable_name+' is not an EPW variable.')
        print('Valid EPW variables:')
        print(field_names[6:])
        raise SystemExit
    else:
        print(epw_variable_name+' is an EPW variable')

# ----------------------------------------------------------------


def check_epw_inputs(location_name: str,
                     epw_read: str, epw_write: str,
                     lon: float, lat: float,
                     epw_file_name=None):

    if lon is None or lat is None:
        print('Both longitude and latitude coordinates are required.')
        raise SystemExit

    if location_name is None:
        print('A location name is required')
        raise SystemExit

    if epw_read is None or epw_write is None:
        print('Both read and write epw locations are required.')
        raise SystemExit

    if epw_file_name is not None:
        epw_file = os.path.basename(epw_file_name)
        epw_split = epw_file.split('_')
        epw0 = epw_split[0] == 'CAN'
        epw1 = epw_split[1] == 'BC'
        epw3 = epw_split[3] == 'CWEC2016.epw'
        if epw0 or epw1 or epw3:
            print('Supplied EPW file has incorrect file name format')
            print('Must be:')
            print('CAN_BC_LOCATION.ID_CWEC2016.epw')
            raise SystemExit

# ----------------------------------------------------------------


def get_short_morph_variable(epw_variable: str):
    """ get_short_morph_variable(str)
        Returns the short name of the epw variable
        Args:
            epw_variable(str): The EPW variable name.
        Returns:
            short_morph_variable: Short EPW variable.
    """

    field_names = {
        'dry_bulb_temperature': 'TAS',
        'dew_point_temperature': 'DWPT',
        'relative_humidity': 'RHS',
        'atmospheric_station_pressure': 'PS',
        'extraterrestrial_horizontal_radiation': 'ETHR',
        'extraterrestrial_direct_normal_radition': 'ETDR',
        'horizontal_infrared_radiation_intensity': 'IR',
        'global_horizontal_radiation': 'GHR',
        'direct_normal_radiation': 'DNR',
        'diffuse_horizontal_radiation': 'DHR',
        'wind_direction': 'WDIR',
        'wind_speed': 'WSPD',
        'total_sky_cover': 'TSC',
        'opaque_sky_cover': 'OSC',
        'snow_depth': 'SND',
        'liquid_precipitation_quantity': 'PR'
        }
    short_morph_variable = field_names.get(epw_variable,
                                           'Missing EPW Variable')
    return(short_morph_variable)

# ----------------------------------------------------------------


def add_morphing_info(headers: str, epw_variable: str):
    """ morphing_variable_name(str, str)

        Adds the short name of the epw variable to
        the header information

        Args:
            headers(str): EPW File headers
            epw_variable(str): The EPW variable name.
        Returns:
            morph_headers(str): Headers with short morphing
            variable name added
    """
    morph_val = get_short_morph_variable(epw_variable)
    headers_split = headers.split('\n')
    morph_check = 'MORPHED:' in headers_split[0]
    if morph_check:
        headers_split[0] = headers_split[0] + '|' + morph_val
    else:
        headers_split[0] = headers_split[0] + ' Morphed:' + morph_val
    headers_morph = "\n".join(headers_split)
    return(headers_morph)

# ----------------------------------------------------------------


def gen_future_weather_file(location_name: str,
                            lon: float,
                            lat: float,
                            epw_read: str,
                            epw_write: str,
                            epw_variable_name: str,
                            factor: str,
                            rlen: int,
                            prism_files: list,
                            morphing_climate_files: list,
                            epw_file_name=None):

    """ gen_future_weather_file(str, float, float, str, str,
                                int,list, list,str)

        Regenerates the passed epw file into a weather file
        represeting future data.

        Args:
            location_name(str): The name of the EPW location.
            lon (float): EPW Location coordinate
            lat (float): EPW Location coordinate
            epw_read(str): EPW Read Directory
            epw_write(str): EPW Write Directory
            epw_variable_name(str): The path to the future epw file
                                    to create
            factor (str): Averaging type for the morphing parameters
            roll (int): Window for rolling average (1 if not using)
            prism_files(list): Names of the BC PRISM files
            morphing_climate_files(list): Names of the precomputed files
                         to use for simulated values.
            epw_file_name(str): EPW file to use instead of the
                         nearest file if needed.
    """
    # Confirm the supplied inputs are correct
    check_epw_inputs(location_name, epw_read, epw_write,
                     lon, lat, epw_file_name)

    # Confirm Accurate variable name
    check_epw_variable_name(epw_variable_name)

    # Run the offset to obtain the epw_file
    epw_filename = offset_current_weather_file(lon, lat,
                                               location_name,
                                               prism_files,
                                               epw_read,
                                               epw_write,
                                               epw_file_name)

    epw_var_short = get_short_morph_variable(epw_variable_name)
    epw_file_only = os.path.basename(epw_filename)
    suffix = factor + str(rlen) + epw_file_only.replace('CAN_BC', '')
    epw_new_filename = 'MORPHED_' + epw_var_short + '_' + suffix
    epw_output_filename = epw_write + epw_new_filename

    # Get the data from epw file and the headers from the epw.
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)
    morph_headers = add_morphing_info(headers, epw_variable_name)

    # Morph columns of EPW dataframe based on selected options
    # ----------------------------------------------------------
    # Dry Bulb Temperature
    if epw_variable_name == 'dry_bulb_temperature':
        # Separate Tasmax and Tasmin files from the inputs
        alpha_ix = np.array([i for i, gcm in enumerate(
            morphing_climate_files) if 'alpha_tasmax_tasmin' in gcm])
        alpha_tas_files = np.array(
            morphing_climate_files)[alpha_ix]

        delta_ix = np.array([i for i, gcm in enumerate(
            morphing_climate_files) if 'delta_tas_' in gcm])
        delta_tas_files = np.array(morphing_climate_files)[delta_ix]

        epw_dbt_morph = generate_dry_bulb_temperature(
            epw_data[epw_variable_name],
            epw_data['datetime'],
            lon, lat,
            alpha_tas_files,
            delta_tas_files,
            factor, rlen
        )
        epw_data[epw_variable_name] = epw_dbt_morph
        write_epw_data(epw_data, morph_headers, epw_output_filename)

        print('Successfully morphed dry bulb temperature')
        print('Wrote the morphed file here:' + epw_output_filename)
        return(epw_dbt_morph)
    # ----------------------------------------------------------
    # Dewpoint Temperature
    if epw_variable_name == 'dew_point_temperature':
        print('Dewpoint')
        alpha_ix = np.array([i for i, gcm in enumerate(
            morphing_climate_files) if 'alpha_dewpoint' in gcm])
        alpha_dwpt_files = np.array(
            morphing_climate_files)[alpha_ix]
        delta_ix = np.array([i for i, gcm in enumerate(
            morphing_climate_files) if 'delta_dewpoint_' in gcm])
        delta_dwpt_files = np.array(morphing_climate_files)[delta_ix]

        epw_dwpt_morph = generate_dewpoint_temperature(
            epw_data[epw_variable_name],
            epw_data['datetime'],
            lon, lat,
            alpha_dwpt_files,
            delta_dwpt_files,
            factor, rlen
        )
        epw_data[epw_variable_name] = epw_dwpt_morph

        write_epw_data(epw_data, morph_headers, epw_output_filename)
        print('Successfully morphed dewpoint temperature')
        return(epw_dwpt_morph)
    # ----------------------------------------------------------
    # Horizontal Radiation
    if epw_variable_name == 'global_horizontal_radiation' or \
       epw_variable_name == 'diffuse_horizontail_radiation':
        print('Both Global and Diffuse Horizontal Radiation Series')
        alpha_ix = np.array([i for i, gcm in enumerate(
            morphing_climate_files) if 'alpha_rsds' in gcm])
        alpha_rsds_files = np.array(
            morphing_climate_files)[alpha_ix]

        epw_hr_morph = generate_horizontal_radiation(
            epw_data['global_horizontal_radiation'],
            epw_data['diffuse_horizontal_radiation'],
            epw_data['datetime'],
            lon, lat,
            alpha_rsds_files,
            factor, rlen
        )
        epw_data['global_horizontal_radiation'] = epw_hr_morph[:, 0]
        epw_data['diffuse_horizontal_radiation'] = epw_hr_morph[:, 1]
        write_epw_data(epw_data, morph_headers, epw_output_filename)
        print('Successfully morphed global and diffuse horizontal radiation')
        return(epw_hr_morph)
    # ----------------------------------------------------------
    # Variables Morphed by stretch only
    stretch_variables = ['direct_normal_radiation',
                         'atmospheric_station_pressure',
                         'relative_humidity',
                         'wind_speed',
                         'total_sky_cover',
                         'opaque_sky_cover',
                         'liquid_precipitation_quantity']

    if epw_variable_name in stretch_variables:
        cdf_vars = {'direct_normal_radiation': 'clt',
                    'atmospheric_station_pressure': 'psl',
                    'relative_humidity': 'rhs',
                    'wind_speed': 'wspd',
                    'total_sky_cover': 'clt',
                    'opaque_sky_cover': 'clt',
                    'liquid_precipitation_quantity': 'pr'}

        morphing_functions = {
            'direct_normal_radiation': morph_direct_normal_radiation,
            'atmospheric_station_pressure': morph_atmospheric_station_pressure,
            'relative_humidity': morph_relative_humidity,
            'wind_speed': morph_wind_speed,
            'total_sky_cover': morph_total_sky_cover,
            'opaque_sky_cover': morph_opaque_sky_cover,
            'liquid_precipitation_quantity': morph_liquid_precip_quantity
        }
        cdfvariable = cdf_vars.get(epw_variable_name, 'Missing EPW Variable')

        alpha_ix = np.array([i for i, gcm in enumerate(
            morphing_climate_files) if ('alpha_' + cdfvariable) in gcm])
        alpha_files = np.array(
            morphing_climate_files)[alpha_ix]

        morphing_function = morphing_functions.get(
            epw_variable_name, 'Missing EPW Variable')
        epw_var_morph = generate_stretched_series(epw_data[epw_variable_name],
                                                  epw_data['datetime'],
                                                  lon, lat,
                                                  cdfvariable,
                                                  alpha_files,
                                                  morphing_function,
                                                  factor, rlen)
        epw_data[epw_variable_name] = epw_var_morph
        write_epw_data(epw_data, morph_headers, epw_output_filename)
        print('Successfully morphed '+epw_variable_name)
        print(epw_output_filename)
        return(epw_var_morph)

# -----------------------------------------------------------------


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
    return(rv)

# -----------------------------------------------------------------


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

# -----------------------------------------------------------------


def get_epw_summary_values(input_df: pandas.DataFrame,
                           factor: str, operator: str,
                           rlen: int):
    """ get_epw_summary_values(pandas.DataFrame,str,str,int)

        Calculates averages of daily max, min and average
        from the passed epw data, and returns dataframe of those values.

        Args:
            input_df(pandas.Dataframe): The data and dates read from
                    the epw that we will be averaging with.
            factor(str): The date factor with which to calculate summary
                         statistics
            operator(str): The summary statistic function.
            rlen(int): Averaging window for rolling average

        Returns:
            A dataframe of summary data.
    """
    # ----------------------------------------------------
    aggregate_values = (input_df['data']
                        .groupby(input_df['datetime'].dt.strftime(factor))
                        .agg(operator))
    aggregate_data = aggregate_values.values.flatten()

    # Repeat to maintain 365 morphing values
    if factor == '%m':
        reps = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        aggregate_data = np.repeat(aggregate_data, reps, axis=0)
        aggregate_dates = np.asarray(input_df['datetime'])
    elif rlen != 1:
        agg_series = pandas.Series(aggregate_data)
        agg_roll = agg_series.rolling(rlen, min_periods=1, center=True).mean()
        aggregate_data = np.asarray(agg_roll)
        aggregate_dates = np.asarray(input_df['datetime'])
    else:
        string_dates = [s.replace(" ", "-") for s in aggregate_values.index]
        aggregate_dates = np.asarray([datetime.strptime(date, '%Y-%m-%d')
                                      for date in string_dates])

    agg_dict = {'datetime': aggregate_dates, 'data': aggregate_data}
    return(pandas.DataFrame(agg_dict, columns=['datetime', 'data']))

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

# ---------------------------------------------------------------------


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

# ---------------------------------------------------------------------


def find_closest_epw_file(coords, read_dir):
    """ find_closest_epw_file(coords,read_dir)
        Loops through all epw files in the weather file directory,
        produces a list of coordinates for all available files and
        finds the file nearest to the coords.
        Args:
            coords(float,float): The longitude and latitude to
            compare with the weather files.
            read_dir(string): Location for the weather files.
    """
    files = glob.glob(read_dir+'*.epw')
    coord_data = list_of_epw_coordinates(files)
    wx_index = np.sum(np.square(coord_data-coords), axis=1).argmin()
    wx_selected = files[wx_index]
    return(wx_selected)

# ---------------------------------------------------------------------


def prism_ncfile(varname: str, prism_files: list):
    """ prism_ncfile(varname,prism_files)
        Returns an opened netcdf object for a PRISM climatology file.
        Args:
            varname(str): Variable name
            prism_dir(list): PRISM files
    """
    for i, f in enumerate(prism_files):
        if varname in f:
            ix = i
    fname = prism_files[ix]
    dst = cdf.Dataset(fname, 'r')
    return(dst)

# ---------------------------------------------------------------------


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
    rv = [lat_index, lon_index]
    return(rv)

# ---------------------------------------------------------------------


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

# ---------------------------------------------------------------------


def prism_tas(nc, cells, prism_files):
    """ prism_tas(nc, cells, varname)
        Returns the monthly mean temperature PRISM climatologies
        from the PRISM netcdf file for the provided cell indices.
        Args:
            nc (Open PRISM netcdf object)
            cells(int,int): PRISM cell indices
    """
    ncx = prism_ncfile('tmax', prism_files)
    tmax = prism_read(ncx, cells, 'tmax')
    ncn = prism_ncfile('tmin', prism_files)
    tmin = prism_read(ncn, cells, 'tmin')
    tas = np.divide(tmax + tmin, 2.0)
    return(tas)

# ---------------------------------------------------------------------


def adjust_epw_with_prism(epw_data, prism_diff):
    """ adjust_epw_with_prism(epw_data,prism_diff)
        Adds the PRISM temperature offset to the EPW dry bulb temp and returns
        a new EPW dataframe
        Args:
            epw_data (pandas dataframe): Data from EPW file
            prism_diff(range): 12 monthly PRISM temperature offsets
    """
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    new_epw = epw_data.copy()
    months = range(1, 13)
    for mn in months:
        new_epw.ix[epw_months == mn, 'dry_bulb_temperature'] = round(
            new_epw.ix[epw_months == mn, 'dry_bulb_temperature']
            + prism_diff[mn-1], 1)
    return(new_epw)

# ---------------------------------------------------------------------


def offset_current_weather_file(lon: float,
                                lat: float,
                                location_name: str,
                                prism_files: list,
                                read_dir: str,
                                write_dir: str,
                                epw_filename=None):
    """offset_current_weather_file(float, float,
                                   string,string,
                                   string,string,str)

        Generates an epw file based on a provided location by finding
        the nearest weather file to the supplied coordinates and
        applying an offset to the temperature series based on PRISM
        climatologies (1981-2010 for now).

        Args:
            lat(float): The latitude to read data from climate files.

            lon(float): The logitude to read data from the climate files.

            location_name(string): The name of the coordinate location
                                   supplied.
            prism_files(list): BC PRISM Files.
            read_dir(string): The directory location of the current
                              weather files.
            write_dir(string): The directory location for the new
                              weather files.
            epw_filename(string): Optional EPW file to use instead of
                              the default nearest to coordinates.

    """

    coords = (lon, lat)
    if epw_filename is not None:
        epw_closest = epw_filename
    else:
        # Search through all weather files for the closest to the coords
        epw_closest = find_closest_epw_file(coords, read_dir)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    epw_closest_file = os.path.basename(epw_closest)
    prefix = 'CAN_BC_' + location_name + '_offset_from'
    suffix = epw_closest_file.replace('CAN_BC', '')
    epw_output_name = write_dir + prefix + suffix

    # Return the coordinates of the closest epw file
    epw_closest_coords = get_epw_coordinates(epw_closest)
    # Any PRISM climatology file to grab coordinates
    nc = prism_ncfile('tmax', prism_files)

    prism_cell = get_prism_indices(nc, coords)
    prism_loc_tas = prism_tas(nc, prism_cell, prism_files)

    epw_cell = get_prism_indices(nc, epw_closest_coords)
    prism_epw_tas = prism_tas(nc, epw_cell, prism_files)

    prism_diff = prism_loc_tas - prism_epw_tas
    diff_sum = sum(prism_diff)

    with open(epw_closest) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    if diff_sum == 0:
        print("No offset required")
        write_epw_data(epw_data, headers, epw_output_name)
    else:
        # Get the data from epw file and the headers from the epw.
        epw_offset = adjust_epw_with_prism(epw_data, prism_diff)
        write_epw_data(epw_offset, headers, epw_output_name)

    return(epw_output_name)
