""" gen_future_weather_file.py

	This file takes in climate files from the present (defined as 1951 to 2000) and
	the future (2001 - 2100), as well as an existing epw weather file, and creates a
	new epw file that has projected weather data for a future period.

	The equations and processes outlined in this file are as descibed in the paper
	"Future weather files to support climate resilient building design in Vancouver"
	by Trevor Murdock, published in the 1st international conference on new horizons
	in green civil engineering (NHICE-01), Victoria, BC, Canada, April 25th-27th, 2018.

	Please note that in any documentation, a subscript is represented with an underscore.
	This means that something like dbt_0 actually represents dbt with a subscript 0 following.
	This comvention is NOT the same with variable names, for variable name, an underscore
	simple represents a space (eg hourly_dbt represents hourly dry buld temprature.)
"""

import numpy as np
from datetime import datetime, timedelta
import argparse
import csv
from netCDF4 import Dataset
import netCDF4 as cdf


def calc_dbt(hourly_dbt: float, delta_dbt: float, alpha_dbt: float, daily_dbt_mean: float):
	""" calc_dbt(float, float, float, float)

		calculates the future dry bulb temprature value and returns it.

		Args:
			hourly_dbt(float): The hourly value for drybulb temp, this is
					 the actual value that is being shifted. It
					 is named dbt_0 in the paper referenced above.
			delta_dbt(float): This represents the difference in average dry buld
					  temprature from this day in the future to this day
					  in the present. It is named delta dbt_d in the paper.
			alpha_dbt(float): This is the dividend of the average dry bulb
					  temprature from this day in the future divided by
					  the average dry bulb temprature on this day in the
					  present. It is named alpha dbt_d in the paper.
			daily_mean_dbt(float): This is the mean dry buld temprature for the day
					       that the hourly value was measured in. In the paper
					       this is named <dbt_0>_d

		Returns:
			dbt_0 + delta dbt_d + alpha dbt_d * (dbt_0 - <dbt_0>_d)
			ie, it returns the future dry bulb temprature value for this
			hour. This process is outlined in page 3 of the paper referenced.
	"""

	return hourly_dbt + delta_dbt + alphd_dbt * (hourly_dbt - daily_dbt_mean)

def morph_data(data: list, date: datetime, present_data: list, future_data: list, daily_averages: list):
	""" morph_data(list, datetime, list, list, list)

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
			present_data(list): A list of the data found in the present climate file.
			future_data(list): A list of the data found in the future climate file.
			daily_averages(list): A list of the <x_0>_d values calculatd.

		Returns:
			a list of future data created based off the present data.
	"""

	#: Not yet implemented, so just return the original data to test
	#: that a file will be put together properly.
	return data

def get_epw_data(epw_file: str):
	""" get_epw_data(str)

		Gets the data out of an epw file, and returns it.

		Args:
			epw_file(str): A string to the path of the epw file to read.

		Returns:
			(list, list): The first list returned is a list of lists where each
				      inner list is a row of data from the epw file.
				      The second list is another list of lists where each
				      inner list is a row of header data from the epw file.
			This method will return the data and then the headers in the file.
	"""

	#: Read the epw file as a csv, and convert that csv into a list of strings.
	#: The headers dont need to be changed from here, however the data should be converted
	#: into its specific types.
	with open(epw_file, "r") as epw:
		reader = csv.reader(epw)  #: Read the epw file as a csv
		csv_data = list(reader)  #: Convert the csv reader into a list.
		data = csv_data[8:]  #: Grab the data from the file (rows 8 and on.)
		headers = csv_data[:8]  #: Grab the headers from the file (rows 0 to 7.)

	#: Convert each cell in each row of data to its specific type. This way they can
	#: be changed as required by later functions.
	for index, row in enumerate(data):

		#: Get the date of this row so that we can turn it into a datetime object for
		#: easier use later.
		row_dates = [int(cell) for cell in row[:5]]
		row_time = datetime(row_dates[0], row_dates[1], row_dates[2], row_dates[3] - 1, row_dates[4])

		#: Cells 6 to -9 (9 back from the end) can easilly be just sorted into ints
		#: or floats (float if there's a . in the string, int otherwise.) However, at
		#: position -9 we have a series of numbers representing weather codes. This must be
		#: left as a string.
		row_vals = [float(cell) if "." in cell else int(cell) for cell in row[6:-9]]

		row_vals.append(row[-8])  #: Leave -8 as a string an append it to row_vals.

		#: The rest of the values in the file can be safely converted to ints or floats
		#: without loosing any key information.
		for cell in row[-7:]:
			if "." in cell:
				row_vals.append(float(cell))
			else:
				row_vals.append(int(cell))

		#: Actually change the row we just worked on to have the values we created above.
		data[index] = [row_time, row[5]]
		for cell in row_vals:
			data[index].append(cell)

	#: Return the data and the headers.
	return data, headers

def write_epw_data(data: list, headers: list, filename: str):
	""" write_epw_data(list, list, str)

		Combines the passed headers and data into an epw file with
		the name filename.

		Args:
			data(list): A list of lists, each inner list is a row of
				    data to be written to the epw file.
			header(list): A list of lists, each inner list is a row of
				      the header to be written to the epw file.
			filename(str): The name of the file to be written.
	"""

	#: Reassemble the headers into one string. Each inner list of headers
	#: gets its own line, and each item in each inner list is comma seperated.
	for index, header in enumerate(headers):
		headers[index] = ",".join(header)
	headers = "\n".join(headers)

	#: Rebuild the list of data into a string so that we can write it to a file.
	epw_file = headers + "\n"
	for data_row in data:

		#: epw files mandate that if the -3rd position is 999.000 that it is missing.
		#: This is an issue because 999.000 is not a valid float, as the trailing 0's
		#: are omitted. However, we can assume that if the value is 999.0, that it is
		#: missing, and therefore we should add a string of 999.000 rather than 999.0.
		if data_row[-3] == 999.0:
			data_row[-3] = "999.000"
		#: The same logic as above applies, except with 0.0 and 0.0000.
		if data_row[-6] == 0:
			data_row[-6] = "0.0000"

		#: Get the date that this row is on, and assemble that into the first 5 entries
		#: of the row.
		row_date = data_row[0]
		csv_row = [str(row_date.year), str(row_date.month), str(row_date.day), str(row_date.hour), str(row_date.minute)]

		#: Afterwards, append strings of each cell to the csv_row list so that we have a
		#: list of the exact strings that we want written into this line of the epw file.
		for cell in data_row[1:]:
			csv_row.append(str(cell))

		#: Finally, write that list to the epw_file string (and seperate each entry in the
		#: list with a comma).
		epw_file += ",".join(csv_row) + "\n"

	#: Write the generated string to the passed file.
	#: We pre-generate the string as it is much quicker to append to a string than it is
	#: to write to a file.
	with open(filename,"w+") as epw:
		epw.write(epw_file)


def get_climate_data(climate_file: str, lat: float, long: float, time_range: list):
	""" get_climate_data(str, float, float, list)

		Gets a list of data for each day of each year within time_range from
		the climate file where the location is closest to lat, long.

		Args:
			climate_file(str): The path to the climate file to read from.
			lat(float): The latitude to read data from.
			long(float): The longitude to read data from.
			time_range(Range): The range of years to read data from, to.

		Returns:
			a list with 366 lists. One for each day of the year in a leap year.
			each list contains all the data found in the file for that specific day,
			with each entry in the inner list is 1 year's entry for that day.
	"""

	#: Read the climate data out of the climate file.
	raw_data = Dataset(climate_file)

	#: Get a list of the dates in the climate file.
	data_dates = cdf.num2date(raw_data["time"][:], raw_data["time"].units)

	#: Get the latitude of each location in the file.
	lat_data = raw_data.variables["lat"][:]

	#: Get the logitude of each location in the file.
	long_data = raw_data.variables["lon"][:]

	#: Find the incides of the data with the closest lat and long to those passed.
	lat_index = lat_data.tolist().index(lat_data[np.abs(lat_data - lat).argmin()])
	long_index = long_data.tolist().index(long_data[np.abs(long_data - long).argmin()])

	#: Get the key where the data is located
	data_key = [k for k in raw_data.variables.keys() if k != "lon" and k != "lat" and k != "time"][0]

	#: Grab the actual relevant data from the file.
	data = [[] for _ in range(366)]
	for index, datapoint in enumerate(raw_data.variables[data_key]):
		time = data_dates[index]

		if time.year in time_range:

			#: Appends the datapoint for the lat and long to its day's index in the list.
			#: FIXME: leap year issue.
			#:
			#:	It is probable that time.timetuple().tm_yday wont return 366 for Feb 29th.
			#:	If this is the case then a new function needs to be written to replace
			#:	time.timetuple().tm_yday that returns 366 for Feb 29th, and all occurrences
			#: 	of time.timetuple().tm_yday need to be replaced with that function.
			#:	This is an issue because if feb 29th is not day 366, then all days after
			#: 	feb 29th in a leap year will be shifted by one, causing issues with averaged etc.
			data[time.timetuple().tm_yday - 1].append(datapoint[lat_index][long_index])

	return data

def avg(lst: list):
	""" This method returns the average value of a passed list."""

	return sum(lst) / len(lst) if len(lst) != 0 else 0

def gen_future_weather_file(lat: float,
			    long: float,
			    present_range: Range,
			    future_range: Range,
			    present_climate: str,
			    future_climate: str,
			    epw_file: str
			):
	""" gen_future_weather_file(float, float, Range, Range, str, str, str)

		Regenerates the passed epw file into a weather file represeting future data.

		Args:
			lat(float): The latitude to read data from climate files.
			long(float): The logitude to read data from teh climate files.
			present_range(Range): The range of years that makes up "present" for
					      this particular run.
			future_range(Range): The range of years that makes up "future" for
					     this particular run.
			present_climate(str): The path to the climate file with "present" data.
			future_climate(str): The path to the climate file with "future" data.
			epw_file(str): The path to the epw file to regenerate.
	"""

	#: Get the present and future climate data.
	present_data = get_climate_data(present_climate, lat, long, present_range)
	future_data = get_climate_data(future_climate, lat, long, future_range)

	#: Get the data from epw file and the headers from the epw.
	epw_data, headers = get_epw_data(epw_file)

	#: Morph the data in the file so that it reflects what it should be in the future.
	#: IE) run the processes required in by the paper.
	for row_index, data_row in enumerate(epw_data):
		day = data_row[0].timetuple().tm_yday - 1

		#: morph_data currently has no implementation, so the data being passed
		#: right now may not accurately reflect what will actually be passed upon
		#: implementation.
		epw_data[row_index] = morph_data(data_row, day, present_data, future_data, [])

	#: Write the data out to the epw file.
	write_epw_data(epw_data, headers, epw_file[:-4] + "_future.epw")

if __name__ == "__main__":

	#: Please note, each of these hard coded values is a tester that should
	#: in the future be read from a command line argument rather than hard
	#: coded.
	lat = 49.116
	long = -122.9249
	present_start = 1951
	present_end = 2000
	future_start = 2050
	future_end = 2100
	present_climate = "climate_files/tasmin_gcm_prism_BCCAQ_CNRM-CM5_rcp85_r1i1p1_1951-2000.nc"
	future_climate = "climate_files/tasmin_gcm_prism_BCCAQ_CNRM-CM5_rcp85_r1i1p1_2001-2100.nc"
	epw = "weather_files/CAN_BC_KELOWNA_1123939_CWEC.epw"

	gen_future_weather_file(
	     lat,
	     long,
	     range(present_start, present_end +1),
	     range(future_start, future_end + 1),
	     present_climate,
	     future_climate,
	     epw
	)

