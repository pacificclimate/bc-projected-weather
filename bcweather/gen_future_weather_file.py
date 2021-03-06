"""gen_future_weather_file.py

    This script takes in climate files from the present (defined as
    1951 to 2000) and the future (2001 - 2100), as well as an existing
    epw weather file, and creates a new epw file that has projected
    weather data for a future period.

    The equations and processes outlined in this file are as descibed in the
    paper "Future weather files to support climate resilient building design
    in Vancouver" by Trevor Murdock, published in the 1st international
    conference on new horizons in green civil engineering (NHICE-01),
    Victoria, BC, Canada, April 25th-27th, 2018.

    Please note that in any documentation, a subscript is represented with an
    underscore. This means that something like dbt_0 actually represents dbt
    with a subscript 0 following. This comvention is NOT the same with
    variable names, for variable name, an underscore simple represents a space
    (eg hourly_dbt represents hourly dry buld temprature.)

"""

from argparse import ArgumentParser

from bcweather import gen_future_weather_file

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('lat', type=float, default=49.116,
                        help="Latitude at which to generate a weather file")
    parser.add_argument('lon', type=float, default=-122.9249,
                        help="Longitude at which to generate a weather file")
    parser.add_argument('present_start', type=int, default=1951,
                        help="Starting year of the present climate period")
    parser.add_argument('present_end', type=int, default=2000,
                        help="Ending year of the present climate period")
    parser.add_argument('future_start', type=int, default=2050,
                        help="Starting year of the future climate period")
    parser.add_argument('present_end', type=int, default=2100,
                        help="Ending year of the future climate period")
    parser.add_argument('present_climate_file', type=str,
                        default="climate_files/tasmin_gcm_prism_BCCAQ_"
                        "CNRM-CM5_rcp85_r1i1p1_1951-2000.nc",
                        help="NetCDF file containing the present climate data")
    parser.add_argument('future_climate_file', type=str,
                        default="climate_files/tasmin_gcm_prism_BCCAQ_"
                        "CNRM-CM5_rcp85_r1i1p1_2001-2100.nc",
                        help="NetCDF file containing the future climate data")
    parser.add_argument('netcdf_variable', type=str,
                        default='tasmin',
                        help="Variable name for which to generate a "
                             "weather file")
    parser.add_argument('epw', type=str,
                        default="weather_files/"
                        "CAN_BC_KELOWNA_1123939_CWEC.epw",
                        help="Input weather file for present climate")
    parser.add_argument('epw_output', type=str,
                        default="weather_files/"
                        "CAN_BC_KELOWNA_1123939_CWEC_future.epw",
                        help="Output weather file for future climate")

    args = parser.parse_args()

    gen_future_weather_file(
        args.lat,
        args.lon,
        range(args.present_start, args.present_end + 1),
        range(args.future_start, args.future_end + 1),
        args.present_climate_file,
        args.future_climate_file,
        args.netcdf_variable,
        args.epw,
        args.epw_output
        )
