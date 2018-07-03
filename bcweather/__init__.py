from datetime import datetime

import numpy as np
from netCDF4 import Dataset
import netCDF4 as cdf
from typing import IO
import pandas
import glob

from .epw import epw_to_data_frame

def morph_dry_bulb_temperature(epw_data: pandas.DataFrame,
                               lon: float,lat: float,
                               gcm: str,
                               present_range: range,
                               future_range: range
                              ):
    """ morph_dry_bulb_temperature(pandas.DataFrame, float,float,str,range,range)

        This function takes in data from the epw file, and returns 
        the dataframe with the dry bulb temperature column having been
        replaced with future ("morphed") versions of dry bulb temperature.

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
    ## Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    epw_monthly_averages = get_monthly_values(epw_data['dry_bulb_temperature'],epw_data['datetime'])
    gcm_dir = "/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/" 

    present_tasmax_file = gcm_dir + gcm + "/tasmax_day_BCCAQ2_" + gcm + "_rcp85_r1i1p1_1951-2000.nc"
    future_tasmax_file = gcm_dir + gcm + "/tasmax_day_BCCAQ2_" + gcm + "_rcp85_r1i1p1_2001-2100.nc"
    with Dataset(present_tasmax_file) as f:
        present_tasmax = get_climate_data(f, lat,lon, "tasmax", present_range)
    with Dataset(future_tasmax_file) as f:
        future_tasmax = get_climate_data(f, lat, lon,"tasmax", future_range)
    delta_tasmax = future_tasmax - present_tasmax

    present_tasmin_file = gcm_dir + gcm + "/tasmin_day_BCCAQ2_" + gcm + "_rcp85_r1i1p1_1951-2000.nc"
    future_tasmin_file = gcm_dir + gcm + "/tasmin_day_BCCAQ2_" + gcm + "_rcp85_r1i1p1_2001-2100.nc"
    with Dataset(present_tasmin_file) as f:
        present_tasmin = get_climate_data(f, lat,lon, "tasmin", present_range)
    with Dataset(future_tasmin_file) as f:
        future_tasmin = get_climate_data(f, lat, lon,"tasmin", future_range)
    delta_tasmin = future_tasmin - present_tasmin

    print(present_tasmax_file)
    with Dataset(present_tasmax_file) as f:
        present_std_tasmax = get_climate_data(f, lat,lon, "tasmax", present_range,std='sigma')
    with Dataset(future_tasmax_file) as f:
        future_std_tasmax = get_climate_data(f, lat, lon,"tasmax", future_range,std='sigma')        
    print('Future TX STD')
    print(future_std_tasmax)
    print('Past TX STD')
    print(present_std_tasmax)
    print('TX STD Alpha')
    print(future_std_tasmax/present_std_tasmax)

    delta_tas = (future_tasmax+future_tasmin)/2 - (present_tasmax+present_tasmin)/2
    print('DBT delta')
    print(delta_tas)
    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1,13)
    print('TAS Alpha')
    for mn in months:       
        shift = epw_data.ix[epw_months == mn,'dry_bulb_temperature'] + delta_tas[mn-1]        
        alpha = (delta_tasmax[mn-1] - delta_tasmin[mn-1]) / (epw_monthly_averages['Max'][mn-1] - epw_monthly_averages['Min'][mn-1])
        print(alpha)
        anoms = epw_data.ix[epw_months == mn,'dry_bulb_temperature'] - epw_monthly_averages['Mean'][mn-1]
        morphed_dbt = shift + alpha * anoms
        new_epw.ix[epw_months == mn,'dry_bulb_temperature'] = round(morphed_dbt,1)


    return new_epw


def morph_dewpoint_temperature(epw_data: pandas.DataFrame,
                               lon: float,lat: float,
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
    ## Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    epw_monthly_averages = get_monthly_values(epw_data['dew_point_temperature'],epw_data['datetime'])
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/" 

    ##Dewpoint
    print("Dewpoint")
    dewpoint_alpha_delta = get_ensemble_alpha_and_delta(cdfvariable='dewpoint',lon=lon,lat=lat,
                                                        gcms=gcms,gcm_dir=gcm_dir,ds_type="day",
                                                        present_suffix="_19500101-21001231.nc",
                                                        future_suffix="_19500101-21001231.nc",
                                                        present_range=present_range,future_range=future_range)

    dewpoint_std = get_ensemble_alpha_and_delta(cdfvariable='dewpoint',lon=lon,lat=lat,
                                                gcms=gcms,gcm_dir=gcm_dir,ds_type="day",
                                                present_suffix="_19500101-21001231.nc",
                                                future_suffix="_19500101-21001231.nc",
                                                present_range=present_range,future_range=future_range,
                                                std='sigma')
    print('Delta')
    print(dewpoint_alpha_delta['delta'])
    print('Alpha')
    print(dewpoint_std['alpha'])

    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1,13)
    for mn in months:       
        shift = epw_monthly_averages['Mean'][mn-1] + dewpoint_alpha_delta['delta'][mn-1]        
        stretched_anoms = (epw_data.ix[epw_months == mn,'dew_point_temperature'] - epw_monthly_averages['Mean'][mn-1]) * dewpoint_std['alpha'][mn-1]
        morphed_dewpoint = shift + stretched_anoms
        new_epw.ix[epw_months == mn,'dew_point_temperature'] = round(morphed_dewpoint,1)
    return new_epw

def stretch_gcm_variable(epw_data: pandas.DataFrame,
                         lon: float,lat: float,
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
    ## Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/" 

    ##Find the ensemble average value for alpha
    alpha_delta = get_ensemble_alpha_and_delta(cdfvariable=netcdf_variable,lon=lon,lat=lat,
                                               gcms=gcms,gcm_dir=gcm_dir,ds_type="day",
                                               present_suffix="_19500101-21001231.nc",
                                               future_suffix="_19500101-21001231.nc",
                                               present_range=present_range,future_range=future_range)
    alpha_ens = alpha_delta['alpha']
    print('Alpha Ensemble')
    print(alpha_ens)
    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1,13)
    for mn in months:       
        morphed_data = epw_data.ix[epw_months == mn,epw_variable] * alpha_ens[mn-1]
        if epw_variable == "wind_speed":
            morphed_data = round(morphed_data,1)
        else: 
            morphed_data = round(morphed_data,0).astype(int)
        if epw_variable == "relative_humidity":
            morphed_data[morphed_data > 100] = 100            
        if epw_variable == "total_sky_cover":
            morphed_data[morphed_data > 10] = 10
        if epw_variable == "opaque_sky_cover":
            morphed_data[morphed_data > 10] = 10
        new_epw.ix[epw_months == mn,epw_variable] = morphed_data
    return new_epw
    

def morph_radiation(epw_data: pandas.DataFrame,
                    lon: float,lat: float,
                    gcms: list,
                    present_range: range,
                    future_range: range
                    ):
    """ morph_radiation(pandas.DataFrame, float,float,list,range,range)

        This function takes in data from the epw file, and returns 
        the dataframe with the global horizontal, direct normal and diffuse horizontal 
        radiation variable having been replaced with future versions. The GCM data is from
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
    ## Calculate the monthly averages of tasmax, tasmin, tas for the EPW data
    gcm_dir = "/storage/data/climate/downscale/CMIP5/building_code/" 
    
    ##Total Horizontal Solar Radiation
    print("RSDS")
    rsds_alpha_delta = get_ensemble_alpha_and_delta(cdfvariable='rsds',lon=lon,lat=lat,
                                                    gcms=gcms,gcm_dir=gcm_dir,ds_type="day",
                                                    present_suffix="_19500101-21001231.nc",
                                                    future_suffix="_19500101-21001231.nc",
                                                    present_range=present_range,future_range=future_range)
    alpha_rsds = rsds_alpha_delta['alpha']
    print(alpha_rsds)
    ##Cloud Cover    
    print("CLT")
    clt_alpha_delta = get_ensemble_alpha_and_delta(cdfvariable='clt',lon=lon,lat=lat,
                                                   gcms=gcms,gcm_dir=gcm_dir,ds_type="day",
                                                   present_suffix="_19500101-21001231.nc",
                                                   future_suffix="_19500101-21001231.nc",
                                                   present_range=present_range,future_range=future_range)
    alpha_clt = clt_alpha_delta['alpha']
    print(alpha_clt)
    new_epw = epw_data.copy()
    epw_months = pandas.DatetimeIndex(epw_data['datetime']).month
    months = range(1,13)

    for mn in months:       
        global_horiz_rad = epw_data.ix[epw_months == mn,"global_horizontal_radiation"]
        diffuse_rad = epw_data.ix[epw_months == mn,"diffuse_horizontal_radiation"]
        global_horiz_rad[global_horiz_rad == 0] = 1
        diffuse_to_total_ratio = diffuse_rad / global_horiz_rad
        morphed_global_horiz_rad = epw_data.ix[epw_months == mn,"global_horizontal_radiation"] * alpha_rsds[mn-1]
        morphed_diffuse_horiz_rad = morphed_global_horiz_rad * diffuse_to_total_ratio

        new_epw.ix[epw_months == mn,"global_horizontal_radiation"] = round(morphed_global_horiz_rad,0)
        new_epw.ix[epw_months == mn,"diffuse_horizontal_radiation"] = round(morphed_diffuse_horiz_rad,0)

        normal_rad  = epw_data.ix[epw_months == mn,"direct_normal_radiation"]        
        clouds  = epw_data.ix[epw_months == mn,"total_sky_cover"]
        opaque  = epw_data.ix[epw_months == mn,"opaque_sky_cover"]
        morphed_clouds = clouds * alpha_clt[mn-1]
        morphed_clouds[clouds > 10] = 10
        morphed_opaque = opaque * alpha_clt[mn-1]
        morphed_opaque[opaque > 10] = 10

        ##Assume direct normal radiation responds inversely proportionally to cloud cover changes
        morphed_normal_rad = normal_rad / alpha_clt[mn-1]
        
        new_epw.ix[epw_months == mn,"total_sky_cover"] = round(morphed_clouds,0).astype(int)
        new_epw.ix[epw_months == mn,"opaque_sky_cover"] = round(morphed_opaque,0).astype(int)
        new_epw.ix[epw_months == mn,"direct_normal_radiation"] = round(morphed_normal_rad,0)
        
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
    """ write_epw_data(list, list, str)

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
    
    for ix in range(0,data.shape[0]):
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
        data_row[-8] = format(data_row[-8],'08d')

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


def get_monthly_values(epw_data: pandas.Series, dates: pandas.Series):
                      ##-> np.ndarray:
    """ get_monthly_values(list)

        Calculates monthly averages of daily max, min and average from the passed epw data, and returns
        a dataframe of those values.

        Args:
            epw_data(pandas.Series): The data read from the epw that we will be
                    averaging with.
            dates(pandas.Series): The datetime series from the pandas DataFrame

        Returns:
            A dataframe of data averaged by month.
    """
    ##print(epw_data)
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
    print(monthly_df)
    return monthly_df

def get_ensemble_alpha_and_delta(cdfvariable: str,
                                 lon: float,
                                 lat: float,                                 
                                 gcms: list,
                                 gcm_dir: str,
                                 ds_type: str,
                                 present_suffix: str,
                                 future_suffix: str,
                                 present_range: range,
                                 future_range: range,
                                 std=None
                                 ):
    """ get_ensemble_alpha_and_delta(cdfvariable,lat,lon,gcms,gcm_dir,ds_type,
                                 present_suffix,future_suffix,present_range,future_range,std)

        Returns the monthly "shift" (delta) and "stretch" (alpha) factors for each month
        from the ensemble averages of the specified netcdf files.
        Args:
            gcms(list): Ensemble of GCMs to use
            ds_type(str): Either "day" or "day_BCCAQ2" if downscaled or gcm data
            *_suffix: The file endind style, differs between downscaled and GCM
            lat(float): The latitude to read data from.
            long(float): The longitude to read data from.
            cdfvariable(str): The variable to read from the netcdf file.
            time_range(range): The range of years to read data from, to.

        Returns:
            a dict with two numpy arrays with 12 values. One for each month of the year.
    """
    print('Alpha Delta Ens Variable')
    print(cdfvariable)                                 
    ##Find the ensemble average value for alpha
    alpha_matrix = np.zeros((12,len(gcms)))
    delta_matrix = np.zeros((12,len(gcms)))
    for i,gcm in enumerate(gcms):
        run = "r1i1p1"
        if gcm == "MIROC5":
            run = "r3i1p1"
        present_file = gcm_dir + gcm + "/"+cdfvariable+"_"+ds_type+"_"+gcm+"_historical+rcp85_"+run+present_suffix
        with Dataset(present_file) as f:
            present_climate = get_climate_data(f, lat,lon, cdfvariable, present_range,std)
        future_file = gcm_dir + gcm + "/"+cdfvariable+"_"+ds_type+"_"+gcm+"_historical+rcp85_"+run+future_suffix
        with Dataset(future_file) as f:
            future_climate = get_climate_data(f, lat, lon, cdfvariable, future_range,std)
        alpha_matrix[:,i] = (future_climate / present_climate)[:,0]
        delta_matrix[:,i] = (future_climate - present_climate)[:,0]
    alpha_ens = np.mean(alpha_matrix,axis=1)
    delta_ens = np.mean(delta_matrix,axis=1)
    rv = {'alpha':alpha_ens,'delta':delta_ens}
    return rv                                 

def get_climate_data(nc: Dataset,
                     lat: float,
                     long: float,
                     cdfvariable: str,
                     time_range: list,
                     std=None
                     ):
    """ get_climate_data(Dataset, float, float, list)

        Gets a list of data for each day of each year within time_range from
        the climate file where the location is closest to lat, long.

        Args:
            nc(Dataset): An open netCDF4.Dataset object.
            lat(float): The latitude to read data from.
            long(float): The longitude to read data from.
            cdfvariable(str): The variable to read from the netcdf file.
            time_range(range): The range of years to read data from, to.

        Returns:
            a list with 12 lists. One for each day of the year in a leap year.
            Each list contains all the data found in the file for that
            specific day, with each entry in the inner list is 1 year's entry
            for that day.
    """

    # Get a list of the dates in the climate file.
    data_dates = cdf.num2date(nc["time"][:], nc["time"].units)

    startyear, endyear = time_range
    t0 = np.argwhere(data_dates >= datetime(startyear, 1, 1)).min()
    tn = np.argwhere(data_dates <= datetime(endyear, 12, 31)).max()

    # Get the latitude of each location in the file.
    lat_data = nc.variables["lat"][:]

    # Get the logitude of each location in the file.
    long_data = nc.variables["lon"][:]

    # Find the incides of the data with the closest lat and long to
    # those passed.
    lat_index = np.absolute(lat_data - lat).argmin()
    long_index = np.absolute(long_data - long).argmin()
    ## Grab the actual relevant data from the file (tn+1 to fix the indexing)
    data = nc.variables[cdfvariable][t0:tn+1, lat_index, long_index]
    data_dict = {'Time':data_dates[t0:tn+1],'Data':data}
    data_frame = pandas.DataFrame(data_dict,columns=['Time','Data'])
    data_monthly = data_frame.groupby(data_frame['Time'].dt.month).mean().values
    if (std=='sigma'):
        data_monthly = data_frame.groupby(data_frame['Time'].dt.month).std().values
    return data_monthly

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

    ##Get the coordinates from the weather file
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

    ##Morph columns of EPW dataframe based on selected options
    epw_dbt_morph = morph_dry_bulb_temperature(epw_data,lon,lat,'MRI-CGCM3',present_range,future_range)
    epw_dpt_morph = morph_dewpoint_temperature(epw_dbt_morph,lon,lat,gcms,present_range,future_range)
    print('Relative Humidity')
    epw_rhs_morph = stretch_gcm_variable(epw_dpt_morph,lon,lat,"relative_humidity","rhs",gcms,present_range,future_range)
    print('Air Pressure')
    epw_psl_morph = stretch_gcm_variable(epw_rhs_morph,lon,lat,"atmospheric_station_pressure","psl",gcms,present_range,future_range)
    print('Windspeed')
    epw_wspd_morph = stretch_gcm_variable(epw_psl_morph,lon,lat,"wind_speed","wspd",gcms,present_range,future_range)
    ##print('Total Sky Cover')
    ##epw_tsc_morph = stretch_gcm_variable(epw_wspd_morph,lon,lat,"total_sky_cover","clt",gcms,present_range,future_range)
    ##print('Opaque Sky Cover')
    ##epw_osc_morph = stretch_gcm_variable(epw_tsc_morph,lon,lat,"opaque_sky_cover","clt",gcms,present_range,future_range)    
    print('Radiation')
    epw_rad_morph = morph_radiation(epw_wspd_morph,lon,lat,gcms,present_range,future_range)

    ##print(monthly_averages.shape)
    # Morph the data in the file so that it reflects what it should be in the
    # future. IE) run the processes required in by the paper.
    # TODO: use the future data to adjust each column of epw_data across time
    # epw_data[column_of_interest] = morph_data(present_data, future_data,
    # monthly_averages)

    # Write the data out to the epw file.
    write_epw_data(epw_rad_morph, headers, epw_output_filename)
    return(epw_output_filename)
##---------------------------------------------------------------------
##---------------------------------------------------------------------
##PRISM offset weather files

def get_epw_coordinates(filename):
    """ get_epw_coordinates(filename)
        Opens the epw file to obtain the first row and extracts the coordinates
        for the epw file
        Args:
            filename(str): An epw filename
    """
    nc = pandas.read_csv(filename,sep=',',header=None,nrows=1)
    rv = (float(nc[7].values), float(nc[6].values)) ##FIXME Replace with pattern matching                                                                  
    return(rv)


def list_of_epw_coordinates(files):
    """ list_of_epw_coordinates(files)
        Obtains the spatial coordinates for all supplied epw files
        Args:
            files(list): A list of all available epw files
    """
    coords = np.zeros((len(files),2))
    for i, file in enumerate(files):
        b = get_epw_coordinates(file)
        coords[i,:] = b
    return(coords)


def find_closest_epw_file(coords):
    """ find_closest_epw_file(coords)
        Loops through all epw files in the weather file directory, produces a list of 
        coordinates for all available files and finds the file nearest to the coords
        Args:
            coords(float,float): The longitude and latitude to compare with the weather files.
    """
    print(coords)
    read_dir = "/storage/data/projects/rci/weather_files/wx_files/" ##FIXME with non-hard coded location
    files = glob.glob(read_dir+'*.epw')
    
    coord_data = list_of_epw_coordinates(files)
    wx_index = np.sum(np.square(coord_data-coords),axis=1).argmin()
    wx_selected = files[wx_index]
    return(wx_selected)

def prism_ncfile(varname):
    """ prism_ncfile(varname)
        Returns an opened netcdf object for a PRISM climatology file.
        Args:
            varname(str): Variable name
    """
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
    rv = [lon_index,lat_index]
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
    return(data[0:12,]) ##13 entries, do not need the annual climatology (13)

def prism_tas(nc,cells):
    """ prism_tas(nc, cells, varname)
        Returns the monthly mean temperature PRISM climatologies from the PRISM netcdf file
        for the provided cell indices
        Args:
            nc (Open PRISM netcdf object)
            cells(int,int): PRISM cell indices
    """    
    ncx = prism_ncfile('tmax')
    tmax = prism_read(ncx,cells,'tmax')
    ncn = prism_ncfile('tmin')
    tmin = prism_read(ncn,cells,'tmin')
    tas = np.divide(tmax + tmin,2.0)
    return(tas)


def adjust_epw_with_prism(epw_data,prism_diff):
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
    months = range(1,13)
    ##FIXME the SetWithCopyWarning for this assignment
    for mn in months:
        ##new_epw.dry_bulb_temperature[epw_months == mn] = round(epw_data.dry_bulb_temperature[epw_months == mn] + prism_diff[mn-1],1)
        new_epw.ix[epw_months == mn,'dry_bulb_temperature'] = round(new_epw.ix[epw_months == mn,'dry_bulb_temperature'] + prism_diff[mn-1],1)
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
    ##Search through all weather files for the closest to the coords
    epw_closest = find_closest_epw_file(coords)

    ##Return the coordinates of the closest epw file
    epw_closest_coords = get_epw_coordinates(epw_closest)
    print(epw_closest_coords)
    ##Any PRISM climatology file to grab coordinates
    nc = prism_ncfile('tmax') 

    print('Closest PRISM cell to supplied coords')
    prism_cell = get_prism_indices(nc,coords)
    print(prism_cell)
    prism_loc_tas = prism_tas(nc,prism_cell)

    print('PRISM coords of cell closest to EPW File')
    epw_cell = get_prism_indices(nc,epw_closest_coords)
    print(epw_cell)
    prism_epw_tas = prism_tas(nc,epw_cell)

    prism_diff = prism_loc_tas - prism_epw_tas
    print(prism_diff)

    # Get the data from epw file and the headers from the epw.
    with open(epw_closest) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)
    
    epw_offset = adjust_epw_with_prism(epw_data,prism_diff)
    print(epw_offset.shape)
    # Write the data out to the epw file.
    epw_output_filename = "/storage/data/projects/rci/weather_files/wx_files/TEST.epw"
    write_epw_data(epw_offset, headers, epw_output_filename)
