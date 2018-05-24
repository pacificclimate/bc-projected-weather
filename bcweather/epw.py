'''
'''

# From: https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html#field-wind-direction # noqa

# These are the names of the fields in a weather file for
# posterity. But, it turns out that we don't care about most of them
# except for the temporal fields (year, month, day, etc.) and
# temperature and precipiation related fields

field_names = (
    'year', 'month', 'day', 'hour', 'minute', 'dry_bulb_temperature',
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
