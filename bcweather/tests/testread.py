import datetime
import glob
import numpy as np
import netcdf
from netCDF4 import Dataset
import netCDF4 as cdf
import pandas as pd
from StringIO import StringIO


def get_epw_coordinates(filename):
    nc = pd.read_csv(filename, sep=',', header=None, nrows=1)
    # FIXME Replace with pattern matching
    rv = (float(nc[7].values), float(nc[6].values))
    return(rv)


def list_of_epw_coordinates(files):
    coords = np.zeros((len(files), 2))
    for i, file in enumerate(files):
        b = get_epw_coordinates(file)
        coords[i, :] = b
    return(coords)


def find_closest_wx_file(coords):
    read_dir = "/home/ssobie/Desktop/weather_files/wx_files/"
    files = glob.glob(read_dir+'*.epw')
    coord_data = list_of_epw_coordinates(files)
    # print(coord_data)
    wx_index = np.sum(np.square(coord_data-coords), axis=1).argmin()
    wx_selected = files[wx_index]
    return(wx_selected)


def ncfile(varname):
    fname = '/home/ssobie/Desktop/weather_files/PRISM/' + varname + '_lm_subset.nc'
    dst = cdf.Dataset(fname, 'r')
    return dst


def get_prism_indices(nc, coords):
    lon_data = nc.variables["lon"][:]
    lon_index = np.absolute(lon_data - coords[0]).argmin()
    lat_data = nc.variables["lat"][:]
    lat_index = np.absolute(lat_data - coords[1]).argmin()
    rv = [lon_index, lat_index]
    return(rv)


def prism_read(nc, cells, varname):
    data = nc.variables[varname][:, cells[0], cells[1]]
    return(data[0:12, ])


def wx_epw_read(filename):
    field_names = (
        'year', 'month', 'day', 'hour', 'minute',
        'data_source_and_uncertainty_flags', 'dry_bulb_temperature',
        'dew_point_temperature', 'relative_humidity',
        'atmospheric_station_pressure', 'extraterrestrial_horizontal_radiation',
        'extraterrestrial_direct_normal_radition',
        'horizontal_infrared_radiation_intensity', 'global_horizontal_radiation',
        'direct_normal_radiation', 'diffuse_horizontal_radiation',
        'global_horizontal_illuminance', 'direct_normal_illuminance',
        'diffuse_horizontal_illuminance', 'zenith_luminance', 'wind_direction',
        'wind_speed', 'total_sky_cover', 'opaque_sky_cover', 'visibility',
        'ceiling_height', 'present_weather_observation', 'present_weather_codes',
        'precipitable_water', 'aerosol_optical_depth', 'snow_depth',
        'days_since_last_snowfall', 'albedo', 'liquid_precipitation_depth',
        'liquid_precipitation_quantity'
    )
    missing_values = (99, 99.9, 999, 9999, 99999, 999999)
    wx = pd.read_csv(filename, sep=',',
                     header=8,
                     names=field_names,
                     index_col=False,
                     na_values=missing_values)
    return(wx)


def prism_tas(nc, cells):
    ncx = ncfile('tmax')
    tmax = prism_read(ncx, cells, 'tmax')
    ncn = ncfile('tmin')
    tmin = prism_read(ncn, cells, 'tmin')
    tas = np.divide(tmax + tmin, 2.0)
    return(tas)


def adjust_wx_with_prism(wx_data, prism_diff):
    wx_months = wx_data[['month']]
    months = range(1, 13)
    for mn in months:
        wx_data.dry_bulb_temperature[wx_data.month ==
                                     mn] += round(prism_diff[mn-1], 1)
    return(wx_data)


def get_epw_header(epw_file):
    pos = epw_file.tell()  # Save the current position
    epw_file.seek(0)
    rv = ''.join([epw_file.readline() for _ in range(8)])
    epw_file.seek(pos)  # Reset the stream position
    return(rv)


def write_epw_data(data, headers, filename):
    epw_file = headers
    print(data)
    for data_row in data.iterrows():
        print(data_row)
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

        # Get the date that this row is on, and assemble that into the first
        # 5 entries of the row.
        csv_row = data_row[0]

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


def prism_closest():
    # Input parameters - coordinates
    coords = (-123.2, 49.2)
    wx_closest = find_closest_wx_file(coords)
    print(wx_closest)

    wx_close_coords = get_epw_coordinates(wx_closest)
    print(wx_close_coords)
    nc = ncfile('tmax')  # Any PRISM file to grab coordinates

    print('Closest PRISM cell to supplied coords')
    prism_cell = get_prism_indices(nc, coords)
    print(prism_cell)
    prism_loc_tas = prism_tas(nc, prism_cell)

    print('PRISM coords of cell closest to WX File')
    wx_cell = get_prism_indices(nc, wx_close_coords)
    print(wx_cell)
    prism_wx_tas = prism_tas(nc, wx_cell)

    prism_diff = prism_loc_tas - prism_wx_tas

    print('Local')
    print(prism_loc_tas)
    print('WX')
    print(prism_wx_tas)
    print('Diff')
    print(prism_diff)

    wx_data = wx_epw_read(wx_closest)
    wx_offset = adjust_wx_with_prism(wx_data, prism_diff)

    with open(wx_closest) as epw_file:
        wx_header = get_epw_header(epw_file)

    write_epw_data(wx_offset, wx_header,
                   "/home/ssobie/Desktop/weather_files/wx_files/test.epw")

    return(wx_offset)


def get_epw_header(epw_file):
    # type: (IO) -> str
    """get_epw_header(IO)                                                                                                                                        Extracts the header from an epw file and returns it.                                                                                                       Args:                                                                                                                                                         epw_file(IO): An open epw file                                                                                                                        Returns: (str): A string consisting of the header, usually the                                                                                                first 8 rows of the file                                                                                                                          """
    pos = epw_file.tell()  # Save the current position
    epw_file.seek(0)
    rv = ''.join([epw_file.readline() for _ in range(8)])
    epw_file.seek(pos)  # Reset the stream position
    return rv


def test_get_epw_header():
    my_string = """Line 1
Line 2
More stuff
Still more worthless data
What else could *possibly* in this file?!
Line 7
Line 8
Line 9
"""
    print(my_string)
    f = StringIO(my_string)
    pos = 15
    f.seek(pos)
    rv = get_epw_header(f)
    print(rv)
    assert rv.startswith("Line 1")
    assert rv.endswith("Line 8\n")
    assert len(rv.splitlines()) == 8
    assert f.tell() == pos
