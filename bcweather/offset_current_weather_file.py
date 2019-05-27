"""offset_current_weather_file.py

    This script takes in coordinates for a location and the location
    name. It finds the nearest EPW weather file to the coordinates 
    and adjusts the temperature field in the file based on the 
    climatological difference in temperature between the file and 
    the location based on PRISM climatologies for 1981-2010.

"""

from argparse import ArgumentParser

from bcweather import offset_current_weather_file

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--lon', type=float, default=None,
                        help="Longitude at which to generate a weather file")
    parser.add_argument('--lat', type=float, default=None,
                        help="Latitude at which to generate a weather file")
    parser.add_argument('--location_name', type=str, required=True,
                        default="UVic",                         
                        help="Name of the coordinate location")
    parser.add_argument('--read_dir', type=str, required=True,
                        default="/storage/data/projects/rci/weather_files/wx_files/",                         
                        help="Location of the original weather files")
    parser.add_argument('--write_dir', type=str, required=True,
                        default="/storage/data/projects/rci/weather_files/wx_files/" \
                        + "morphed_files/TEST/",
                        help="Location of the original weather files")

    args = parser.parse_args()

    offset_current_weather_file(
        args.lon,
        args.lat,
        args.location_name,
        args.read_dir,
        args.write_dir
        )
