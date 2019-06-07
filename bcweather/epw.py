'''
'''
import datetime

import pandas

# From: https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html#field-wind-direction # noqa

# These are the names of the fields in a weather file for
# posterity. But, it turns out that we don't care about most of them
# except for the temporal fields (year, month, day, etc.) and
# temperature and precipiation related fields

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

# Set the year to 1999 otherwise the order is incorrect
def date_converter(*args):
    year, month, day, hour, minute = (int(arg) for arg in args)
    hour -= 1
    return datetime.datetime(1999, month, day, hour, minute)


def epw_to_data_frame(file_):
    """ epw_to_data_frame(IO)

        Gets the data out of an epw file, and returns it as a pandas DataFrame.

        Args:
            file_(IO): A filepath or buffer or any object with a read() method

        Returns:
            (pandas.DataFrame): The data includes all 30 columns of data, with
                      NaNs handled and types converted.
    """
    return pandas.read_csv(
        file_,
        header=7,
        names=field_names,
        index_col=False,
        parse_dates={'datetime': [0, 1, 2, 3, 4]},
        date_parser=date_converter
    )

# na_values=missing_values,
