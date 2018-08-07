#!/bin/sh

PYTHONPATH='/storage/home/ssobie/code/repos/bc-projected-weather' 

ds_dir='/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/'

epw_filename='/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw'
lon=0
lat=0
epw_output_filename='/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/BASH_CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw'                 
epw_variable_name='dry_bulb_temperature'
present_start=1971
present_end=2000
future_start=2041
future_end=2070
factor='%m'

rcp='rcp85'
path="${ds_dir}*/"

#present_climate_files= $(find $path -name "tasm*_day_BCCAQ2*${rcp}*1951-2000.nc" -print0 | xargs -0 grep -e "ACCESS1-0" -e "CanESM2") 
#present_climate_files= $(find $path -name "tasm*_day_BCCAQ2*${rcp}*1951-2000.nc" -exec grep -e "ACCESS1-0" -e "CanESM2" {} \;) 
#present_climate_files= $(find $path -name "tasm*_day_BCCAQ2*${rcp}*1951-2000.nc" -print0 | xargs -0 grep -l "(ACCESS1-0|CanESM2)")
present_climate_files= $(find $path -name "tasm*_day_BCCAQ2*${rcp}*1951-2000.nc") 
future_climate_files=$(find $path -name "tasm*_day_BCCAQ2*${rcp}*2001-2100.nc")

#Length of array
#echo ${!present_tasmax_files[@]}

printf '%s\n' "${present_climate_files[@]}"
#printf '%s\n' "${future_climate_files[@]}"

echo $epw_output_filename
echo $epw_variable_name
echo $present_start
echo $present_end
echo $future_start
echo $future_end
#echo $present_climate_files
#echo $future_climate_files
echo $factor
echo $epw_filename
echo $lon
echo $lat

python3 /storage/home/ssobie/code/repos/bc-projected-weather/bcweather/gen_future_weather_file.py $epw_output_filename $epw_variable_name $present_start $present_end $future_start $future_end $present_climate_files $future_climate_files $factor $epw_filename $lon $lat

