from datetime import datetime

import numpy as np
from netCDF4 import Dataset
import netCDF4 as cdf
from typing import IO
import pandas
import glob

from .epw import epw_to_data_frame


def calc_dbt(hourly_dbt: float,
             delta_dbt: float,
             alpha_dbt: float,
             daily_dbt_mean: float
             ):
    """ calc_dbt(float, float, float, float)

        calculates the future dry bulb temprature value and returns it.

        Args:
            hourly_dbt(float): The hourly value for drybulb temp, this is
                      the actual value that is being shifted. It
                      is named dbt_0 in the paper referenced above.

            delta_dbt(float): This represents the difference in average dry
                      bulb temprature from this day in the future to this day
                      in the present. It is named delta dbt_d in the paper.

            alpha_dbt(float): This is the dividend of the average dry bulb
                      temprature from this day in the future divided by
                      the average dry bulb temprature on this day in the
                      present. It is named alpha dbt_d in the paper.

            daily_mean_dbt(float): This is the mean dry buld temprature for
                       the day that the hourly value was measured in. In the
                       paper this is named <dbt_0>_d

        Returns:
            dbt_0 + delta dbt_d + alpha dbt_d * (dbt_0 - <dbt_0>_d)
            ie, it returns the future dry bulb temprature value for this
            hour. This process is outlined in page 3 of the paper referenced.
    """

    return hourly_dbt + delta_dbt + alpha_dbt * (hourly_dbt - daily_dbt_mean)


def morph_data(data: np.array,
               date: datetime,
               present_data: np.array,
               future_data: np.array,
               daily_averages: np.array
               ):
    """ morph_data(numpy.array, datetime, numpy.array, numpy.array, numpy.array)

        This method takes in a single line of data from the epw file,
        and returns that line with the relevant datafields having been
        replaced with future versions of that data (as calculated.)

        Args:
            data(list): A list of data representing the csv line found
                    in the epw file. IE one line of data, that has been
                    turned into a list (rather than a string read directly
                    from the file.)

            date(datetime): A datetime object representing the exact hour that
                    this specific data row is from.

            present_data(list): A list of the data found in the present climate
                    file.

            future_data(list): A list of the data found in the future climate
                    file.

            daily_averages(list): A list of the <x_0>_d values calculatd.

        Returns:
            a list of future data created based off the present data.
    """

    raise NotImplementedError


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


def get_daily_averages(epw_data: pandas.Series, dates: pandas.Series) \
                      -> np.ndarray:
    """ get_daily_averages(list)

        Calculates each day's average from the passed epw data, and returns
        a list of those averages.

        Args:
            epw_data(pandas.Series): The data read from the epw that we will be
                    averaging with.
            dates(pandas.Series): The datetime series from the pandas DataFrame

        Returns:
            A numpy array of data averaged by julian day (day of year).
    """
    return epw_data.groupby(dates.dt.strftime('%m %d')).mean().values


def get_climate_data(nc: Dataset,
                     lat: float,
                     long: float,
                     cdfvariable: str,
                     time_range: list
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
            a list with 366 lists. One for each day of the year in a leap year.
            Each list contains all the data found in the file for that
            specific day, with each entry in the inner list is 1 year's entry
            for that day.
    """

    # Get a list of the dates in the climate file.
    data_dates = cdf.num2date(nc["time"][:], nc["time"].units)

    startyear, endyear = time_range
    t0 = np.argwhere(data_dates >= datetime(startyear, 1, 1)).min()
    tn = np.argwhere(data_dates <= datetime(endyear, 1, 1)).max()

    # Get the latitude of each location in the file.
    lat_data = nc.variables["lat"][:]

    # Get the logitude of each location in the file.
    long_data = nc.variables["lon"][:]

    # Find the incides of the data with the closest lat and long to
    # those passed.
    lat_index = np.absolute(lat_data - lat).argmin()
    long_index = np.absolute(long_data - long).argmin()

    # Grab the actual relevant data from the file.
    data = nc.variables[cdfvariable][t0:tn, lat_index, long_index]
    return data


def gen_future_weather_file(lat: float,
                            long: float,
                            present_range: range,
                            future_range: range,
                            present_climate: str,
                            future_climate: str,
                            netcdf_variable: str,
                            epw_filename: str,
                            epw_output_filename: str
                            ):
    """ gen_future_weather_file(float, float, range, range, str, str, str)

        Regenerates the passed epw file into a weather file represeting future
        data.

        Args:
            lat(float): The latitude to read data from climate files.

            long(float): The logitude to read data from teh climate files.

            present_range(range): The range of years that makes up "present"
                          for this particular run.

            future_range(range): The range of years that makes up "future" for
                         this particular run.

            present_climate(str): The path to the climate file with "present"
                         data.

            future_climate(str): The path to the climate file with "future"
                         data.

            epw_filename(str): The path to the epw file to regenerate.

            epw_output_filename(str): The path to the future epw file to create
    """

    # Get the present and future climate data.
    # with Dataset(present_climate) as f:
    #     present_data = get_climate_data(f, lat,
    #                                     long, netcdf_variable, present_range)

    # with Dataset(future_climate) as f:
    #     future_data = get_climate_data(f, lat, long,
    #                                    netcdf_variable, future_range)

    # Get the data from epw file and the headers from the epw.
    with open(epw_filename) as epw_file:
        epw_data = epw_to_data_frame(epw_file)
        headers = get_epw_header(epw_file)

    # daily_averages = get_daily_averages(epw_data)

    # Morph the data in the file so that it reflects what it should be in the
    # future. IE) run the processes required in by the paper.
    # TODO: use the future data to adjust each column of epw_data across time
    # epw_data[column_of_interest] = morph_data(present_data, future_data,
    # daily_averages)

    # Write the data out to the epw file.
    write_epw_data(epw_data, headers, epw_output_filename)

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
    print(epw_data.loc[0].values.tolist())
    epw_output_filename = "/storage/data/projects/rci/weather_files/wx_files/TEST.epw"
    write_epw_data(epw_offset, headers, epw_output_filename)
