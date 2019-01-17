#!/bin/sh

#---------------------------------------
# EPW Information
read_dir='/storage/data/projects/rci/weather_files/wx_files/'
location_name='Finnerty'
lon=-123.304926
lat=48.469187

echo 'python3 /storage/home/ssobie/code/repos/bc-projected-weather/bcweather/offset_current_weather_file.py --lon' ${lon} '--lat' ${lat} '--location_name' ${location_name} '--read_dir' ${read_dir} 

python3 /storage/home/ssobie/code/repos/bc-projected-weather/bcweather/offset_current_weather_file.py --lon $lon --lat $lat --location_name $location_name --read_dir $read_dir 


