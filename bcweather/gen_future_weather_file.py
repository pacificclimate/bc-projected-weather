"""gen_future_weather_file.py

    This script takes in files of precomputed morphing factors
    as well as an existing epw weather file, and creates a new
    epw file that has projected weather data for a future period.

"""

from argparse import ArgumentParser

from bcweather import gen_future_weather_file

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--location_name', type=str, required=True,
                        default="NEW_SITE",
                        help="Name for the location coordinates")
    parser.add_argument('--lon', type=float, required=True,
                        default=None,
                        help="Longitude at which to generate a weather file")
    parser.add_argument('--lat', type=float, required=True,
                        default=None,
                        help="Latitude at which to generate a weather file")
    parser.add_argument('--epw_read', type=str, required=True,
                        default="/storage/data/projects/rci/weather_files/"
                        + "wx_2016/",
                        help="Read location for the EPW file")
    parser.add_argument('--epw_write', type=str, required=True,
                        default="/storage/data/projects/rci/weather_files/"
                        + "wx_2016/morphed_files/",
                        help="Write location for the EPW file")
    parser.add_argument('--epw_variable_name', type=str, required=True,
                        default='dry_bulb_temperature',
                        help="EPW variable name to morph")
    parser.add_argument('--factor', choices=['monthly', 'daily', 'roll'],
                        required=True, default="monthly", type=str,
                        help="Factor with which to average (daily or monthly")
    parser.add_argument('--rlen', default=1, type=int,
                        help="Averaging window if roll is specified")
    parser.add_argument('--prism_files', action='append',
                        required=True,
                        help="NetCDF files containing the PRISM Climatologies")
    parser.add_argument('--morphing_climate_files', action='append',
                        required=True,
                        help=("NetCDF files containing the "
                              "morphing climate factors"))
    parser.add_argument('--epw_filename', type=str,
                        default=None,
                        help=("Optional input weather file "
                              "to use instead of closest."))

    args = parser.parse_args()

    gen_future_weather_file(
        args.location_name,
        args.lon,
        args.lat,
        args.epw_read,
        args.epw_write,
        args.epw_variable_name,
        args.factor,
        args.rlen,
        args.prism_files,
        args.morphing_climate_files,
        args.epw_filename,

        )
