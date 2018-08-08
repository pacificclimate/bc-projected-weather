#!/bin/sh

ds_dir='/storage/data/climate/downscale/BCCAQ2+PRISM/high_res_downscaling/bccaq_gcm_bc_subset/'

epw_filename='/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw'
#lon
#lat=0

epw_output_filename='/storage/home/ssobie/code/repos/bc-projected-weather/bcweather/tests/data/BASH_CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw'                 
epw_variable_name='dry_bulb_temperature'
present_start=1971
present_end=2000
future_start=2041
future_end=2070
factor='monthly'
rcp='rcp85'

path="${ds_dir}*/"
pf='--present_climate_files'
ff='--future_climate_files'

##For Tasmax/Tasmin
for gcm in "ACCESS1-0" "CanESM2" "CNRM-CM5" "CSIRO-Mk3-6-0" "GFDL-ESM2G" "HadGEM2-CC" "HadGEM2-ES" "inmcm4" "MIROC5" "MRI-CGCM3"; do
    echo $gcm
    past_tx=$(find $path -name "tasmax*_day_BCCAQ2_"${gcm}"_"${rcp}"_*1951-2000.nc")
    past_tn=$(find $path -name "tasmin*_day_BCCAQ2_"${gcm}"_"${rcp}"_*1951-2000.nc")
    present_climate_files=$present_climate_files\ $pf\ ${past_tx}\ $pf\ ${past_tn}

    proj_tx=$(find $path -name "tasmax*_day_BCCAQ2_"${gcm}"_"${rcp}"_*2001-2100.nc")
    proj_tn=$(find $path -name "tasmin*_day_BCCAQ2_"${gcm}"_"${rcp}"_*2001-2100.nc")
    future_climate_files=$future_climate_files\ $ff\ ${proj_tx}\ $ff\ ${proj_tn}
done


#present_climate_files=$(for i in $(find $path -name "tasmax*_day_BCCAQ2*1951-2000.nc"); do echo "--present_climate_files $i"; done)
#future_climate_files=$(for i in $(find $path -name "tasmax*_day_BCCAQ2*2001-2100.nc"); do echo "--future_climate_files $i"; done)

##echo 'Present Filenames'  
##echo $present_climate_files

##echo 'Future Filenames'  
##echo $future_climate_files

#printf '%s\n' "${present_climate_files[0]}"
#printf '%s\n' "${future_climate_files[@]}"
echo ' '

#echo $present_climate_files

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
#echo $lon
#echo $lat

echo 'python3 /storage/home/ssobie/code/repos/bc-projected-weather/bcweather/gen_future_weather_file.py --epw_output_filename' ${epw_output_filename} '--epw_variable_name' ${epw_variable_name} '--present_start' ${present_start} '--present_end' ${present_end} '--future_start' ${future_start} '--future_end' ${future_end} ${present_climate_files} ${future_climate_files} '--factor' ${factor} '--epw_filename' ${epw_filename}

python3 /storage/home/ssobie/code/repos/bc-projected-weather/bcweather/gen_future_weather_file.py --epw_output_filename $epw_output_filename --epw_variable_name $epw_variable_name --present_start $present_start --present_end $present_end --future_start $future_start --future_end $future_end $present_climate_files $future_climate_files --factor $factor --epw_filename $epw_filename 


