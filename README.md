# bc-projected-weather
Tools for generating projected weather statistics for locations in BC.

Handoff notes:

	Current code breakdown and issues:

		As the current code sits, A future climate file, present climat file,
		and weather file is read from a variable. A latitude, and longitude are also
		provided; these values are used to grab specifc datasets from the
		climate files. Next, 4 years are provided; present_start represents
		the first year in the "present" period, and present_end represents the
		last year in the "present" period (similar logic applies to
		future_start, and future_end.) Finally, another variable is provided
		called "netcdf_variable"; this is the name of the variable in the
		climate file where the actual data is stored.

		Current issues with the above:
			1) All variables described above need to be read from command line.
			2) More that 2 climate files may need to be read for each run.
			3) Following the above issue, more that 1 netcdf variable may be needed each run.

		Potential solutions:
			1) Use the argparse library. I just didn't have time to implement.
			2/3) Have each run only morph one variable at a time, and have another
			function actually control this function, parsing command line info and
			controlling each run seperately. There is probably a better way, but this
			was my first thought.

		gen_future_weather_file() takes in the information as described above. It then
		extracts the data from the the provided future and present climate files, Draws
		the information out of the provided epw (weather file), gets daily averages for
		each type of data found in the epw. Once it has its daily averages, it will pass
		all of the found information, and one row of data from the epw into "morph_data".
		As of current, morph_data has not been implemented, and just returns the line of
		data from the epw. morph_data should take the row, and morph each entry into its
		future variant based off of Trevor's paper. Finally, it will write the morphed epw
		data to an epw file.

		Current issues with the above:
			1)  As of current, getting climate data is limited to only 2 weather files,
			one present and one future.
			2) morph data has no current implementation.

		Potential solutions:
			1) Instead of passing in one future and one present climate file,
			pass in a list of each, and then gather data for each in the list.
			2) I haven't had time to think through how to implement this, sorry.

		

	Climate (netCDF) file info:
		Each climate file (netCDF file) is broken down into
		4 variables, time, lat, long, and then a variable (eg tasmin) that
		holds the actual data. 
		The variable then looks somewhat like a list of list of lists

		For each outermost list, we have a list of lists, each outer list represents
		a time at daily resolution. The middle list is a list of lists, where each outer
		lists represents a latitude, and each inner list has the data for that hour for
		that longitude for that latitude. Thats confusing, heres an example:

			if time[0] = 1 (for 1 day from Jan 1st, 1950), lat[5] = 80, and lon[6] = -123
			then tasmin[0][5][6] holds the data for day 1, at latitude 80, and longitude -123

	
Finally, here's all of the emails I had with Trevor describing what the project should do
when it's done, hopefully this will help you get off to a better start.

NOTE: Anything starting with [[TQM]] is a responce from Trevor, the other lines are from me.


[[TQM]] Yes 4 am and 4 pm

 

[[TQM]] And yes just alpha is affected by the interpolation since there is only an alpha for temperature and this only affects temperature

 
------------------------------------------------------------------------------------------------------------------------

aha ok this clears up a lot of questions thank you,

 

I think the only questions I have left then are, how do we know which hour that tasmin and tasmax were from, I dont see that information anywhere in the netcdf file (or do we always assume that it will be at 4am and 4pm...?)

 

Ok, I think the main issue was that I thought that the interpolation was happening as part of calcuating <dtb0>d but if that's just the daily average, then that all makes total sense.

 

and just double checking, that means that the only value being effected by the interpolation is the delta value? not the alpha or (dtb0 - <dtb0>d), yes?

 

Thank you again, I think that's the last of the questions,

 

Ross

------------------------------------------------------------------------------------------------------------------------

Sorry Trevor, I still think I'm missing something here. when you say tasmin and tasmax, are those values that I'm finding from the epw, or the tasmin and tasmax netcdf files? I think what I've gotten from your email is that tasmax and tasmin are just names for the daily high and daily low temperatures of each day.

 

[[TQM]] The tasmin and tasmax netcdf files only. And yes that’s right about the names. In other words from the climate models we have temperature twice daily but in the weather files you have hourly. So we know the change at two hours of the day and we just linearly interpolate between those changes for the other hours.

 

Presuming my thought above is correct, I then understand that I need to take the the hour of the max and min of the day using the epw and find how far away each is from the current hour.

 

Then I take that information about how far each of the max and min hours are from the current hour, and use those as percentages to scale the change I would have made?

 

[[TQM]] Yes that’s what I mean by linear interpolation.

 

If all of that is the case, where am I finding this change that would be made from? you say for example that the nighttime low change is 2 degrees, where am I finding this change from if that's the case? are we no longer calculating <dbt0>d anymore at this point? I think somewhere I've confused when the process you're talking about below is applied.


[[TQM]] It’s the delta.. i.e. the shift factor. So you’re calculating two deltas for dry bulb temperature – one for each of those times of day.

 

Thanks, and sorry for the delay,


[[TQM]] No worries and thanks for getting this question in before I’m completely unavailable Monday/Tuesday.

 

Ross

------------------------------------------------------------------------------------------------------------------------

I’ll explain with an example. Note that we’ll assume tasmin (nighttime low) is at 4 am and tasmax (daytime high) is at 4 pm.

Say the nighttime low change is 2 degrees Celsius and the daytime high change is 1 Celsius.

The change in between is just linearly in between those to values. So ½ way between 4 pm and 4 am (at both 10 am and 10 pm), the change should be 1.5 C

Trevor

------------------------------------------------------------------------------------------------------------------------

Hi Trevor,

 

Thank you for the explanations. You said "for dry bulb temperature, you need to interpolate the change between the daytime high and nighttime low first and apply that interpolated change to each hour." Would you mind describing what this means a little further? As i understand I need to get the warmest value and the coldest value for each day from the weather file, and then "interpolate" them. How do I go about interpolating them (ie what is the mathematical function I'm actually applying to these found values?)

 

Thanks,

 

Ross

------------------------------------------------------------------------------------------------------------------------

I'm almost done making something that will shift the dry bulb temp, I'm taking a bit loner because I'm try to set it up so that it can easily be combined with the other measured values when I'm on to working with those.

[[TQM]] It’s ok if we don’t get to variables other than dry bulb and precipitation before the end of the month. If we do that’s great. In any case I appreciate that you’re taking the time to make the code flexible so that when we pick it up and run with it after, it will be easy for us to modify for the other equations (that’s exactly when I sent the full article for that extra context).

 

I had one more question for you that's come up, when calculating dtb, we have 〈dtb0〉d. I'm not exactly sure how I should go about calculating this value, is it just the average value of day d's dtb in the epw file? or is it the average across all day d's in one of the present or future climate files? or something else?

 

[[TQM]] Yes now you’re getting into combining things. So the delta and shift factors are from the climate model only but both the 〈𝑥0〉𝑑 and the final 〈𝑥〉𝑑 are from the weather file. There’s also the further wrinkle (not really described well in the paper) that there are actually hourly data in both the original and future shifted weather files. In the case of precipitation you essentially just apply the same operation to each hour of each day. For dry bulb temperature, you need to interpolate the change between the daytime high and nighttime low first and apply that interpolated change to each hour.

 

Thanks,

 

Ross

------------------------------------------------------------------------------------------------------------------------

It is spatial data so 171 x 339 are the locations – so it’s like a 2D array of time series. This is very high resolution (800 m) data. The metadata should describe how to find the location. At this point I don’t have a particular location in mind - you’ll just want to match the location for a given weather file to the location you extra from the netcdf file.

 

In other words, you’ll perform the operations to calculate the stretch and shift factors after choosing just one time series from the array of time series. As far as the # days note that some climate models have leap years while others don’t. And one of the models has 360 day years – just to complicate things.

 

Trevor

------------------------------------------------------------------------------------------------------------------------

Hi Trevor,

 

I finally got those files all downloaded (kept having issues with the download failing) and I'm having trouble interpreting the data in the climate files. For each day worth of data is seems as if there is 171 sets of 339 readings? specifically for the file tasmin_gcm_prism_BCCAQ_CNRM-CM5_rcp85_r1i1p1_2001-2100.nc there are 36524 days, and for each of those days it looks like there are 171 sets of 339 readings, so I'm not entirely sure how to interpret that (the time dictionary has 36524 entries, and the tasmin dictionary has 36524 entries, however tasmin has 171 sets of 339 numbers.) Do you have any idea what's going on here?

 

Thanks,

 

Ross

------------------------------------------------------------------------------------------------------------------------

Start with just dry bulb temperature. There are two wrinkles you will encounter. One I mentioned before – that after getting the daily deltas we then smooth them before applying.

 

Second, we come up with a different delta for each hour of the day by interpolating between the nighttime low and daytime high delta for each day.

 

Sorry I should have said that we’re not following the paper exactly here for the purposes of getting the code off the ground. You can ignore everything other than dry bulb temperature. Also we’re not going to be applying the scaling to temperature – JUST the shift.

 

Precip isn’t in the paper but for it we will want to apply a shift.

 

Trevor

------------------------------------------------------------------------------------------------------------------------

Ok I'll do that.

 

Ok, so if that's the case then which equations should I actually be running?

 Dry bulb temp I presume,  then the only other one's I'm seeing that dont involve humidity or radiation are wind speed and cloud cover, Is there an equation here somewhere for precipitation that I'm just not seeing? (And would you like me to calculate the wind speed and cloud cover as well.)

 

Ross

------------------------------------------------------------------------------------------------------------------------

OK I'll download one of each of those. What is in each file? And why do I need all 3? are they each different climate files? or? As I understood there would be one climate file with 150 years worth of data that I should read from, as it appears from those files each climate file is broken into 1951-2000, and then 2001 - 2100. further what do yeach of tasmax, tasmin, and pr contain that is different from each other?


[[TQM]] Ah I had forgotten that they were broken into 1951-2000 and 2001-2100. So you only need the former for the 1971-2000 baseline from which to compute the scaling factors and all the future periods will come from the second file.

 

[[TQM]] tasmax = nighttime low temperature, tasmin = daytime high temperature, and prec = daily precipitation. For now, you can ignore all of the other equations in the paper that depend on things like radiation and humidity, etc.

Sorry for the flurry of questions both before and now,


[[TQM]] No worries – keep them coming. I will be more responsive today and tomorrow. Next week I’m quite tied up so don’t be shy about questions right now!

 

Ross

------------------------------------------------------------------------------------------------------------------------

https://pacificclimate.org/~ssobie/wx_files/

 

I recommend that you start with CNRM-CM5. After that if it’s not too cumbersome to download multiple that would be great if you did but for the purpose of the code even if you only processed one GCM run that would be fine.

 

You can use the ones with “gcm-prism” in the files names. You’ll need 3 files for each model: one each for tasmax, tasmin, and pr

 

Trevor

------------------------------------------------------------------------------------------------------------------------

Thank you very much,

 

And you are correct in your assumption, still no NetCDF files.


Ross

------------------------------------------------------------------------------------------------------------------------

Oh!


For the weather files you can download them from ftp://client_climate@ftp.tor.ec.gc.ca/Pub/Engineering_Climate_Dataset/Canadian_Weather_year_for_Energy_Calculation_CWEC/ENGLISH/CWEC_v_2016/BC_CWEC.zip

 

I’m assuming you still haven’t gotten NetCDF files for the climate projections either. I’ll talk to Steve today.

 

Trevor

------------------------------------------------------------------------------------------------------------------------

Hi Trevor,

 

I figured that I should mention that I still haven't received any weather files yet.

 

Also, could you please explain what's going on with the global horizontal radiation, and the direct normal radiation? more specifically, what is meant by 'using the absolute change between the climate baselines, (delta)rsdsd'? second, does the gsr variable represent Global horizontal radiation rather than ghr? otherwise I'm not sure where gsr is coming from and what it means. Finally, in the equation for direct normal radiation, dnr is not being solved for, should I rearrange that equation so that it is being solved for? again I think that question is coming from my not understanding what gsr is meaning.

 

Thanks,

 

Ross

------------------------------------------------------------------------------------------------------------------------

Aha, that makes a lot more sense than how I had it in my head. Ok, so we take an input of a climate file (containing 150 years worth of daily resolution data.) From this we calculate the delta and alpha values, and averages over both the present, and future ranges for each day in the year. Then we look at the weather file also given as an input, and we run each of the equations found in the paper against each hour's data, using the found averages, deltas, and alphas. Finally, as we run each equation, we build a new netCDF weather file, and we will export it when we're done. Does that sound about right?

[[TQM]] Yes exactly. Note that we use the term “run” to mean something different in climate science so I will avoid that term here but yes this is what is happening.

 

The only other question I have (presuming that the above correctly describes what needs to happen) is should I expect an input as to whether we are looking at the 2011-2040, 2041-2070, or 2071-2100 future time? and if so, what would you want that input to look like?

[[TQM]] Yes it would be good to allow the future period to be a variable. I think ideally we might want the flexibility to specify any start and end date. The standard would be those 3. Similarly, we might want to be able to change the baseline but the default should be 1971-2000.

 

Thank you again,

 

Ross

------------------------------------------------------------------------------------------------------------------------ 

No worries about the delay, stuff happens. Most of that seems to make sense, I just want to clarify. <x>d says to take the average of day d, over the 30 year interval (Seeing as how we have hourly resolution in the files, does this mean that I should average each hour to get a daily average, and then take the 30 of those averages and average them?).

 

[[TQM]] I think you’re still conflating the weather files and the climate data files – they are quite different.

[[TQM]] The weather files have hourly resolution but just a single year.

[[TQM]] The climate data NetCDF files that Steve will set you up with have daily time resolution for 150 years. Everything in the paper is in this “space”. In other words shift and scale corrections are computed entirely from climate model daily data by averaging over 30 years of record. Once they are computed THEN we apply them to the weather files – at which point we have to deal with the fact that the corrections are daily but the weather file has hourly resolution. I’ll explain how we do that below (it’s not in the paper – though once I describe it you should recognize it in the code).

 

Next, does this mean that (as a NetCDF file only has 1 year worth of data) Each run should use 60 files (30 for xpresent and 30 more for xfuture)?

 

[[TQM]] Again that’s backwards. The weather file has only 1 year worth of data. The NetCDF files have 150 years. Each run will use 1 file. You’re averaging over two 30-year periods which are each a subset of the 150 years of record. Also note that the NetCDF files contain gridded data across the whole country – you’ll be picking a single location from these files.

 

Finally, if thats the case, then how should we go about going from daily resolution to hourly resolution (presuming thats a requirement of NetCDF)?


[[TQM]] NetCDF is just a format, it doesn’t have any resolution requirements, but you’re quite right that we have to reconcile the difference between the hourly and daily resolutions. I’ll describe it now for precipitation and just note that for temperature there’s an added wrinkle which I’ll explain later. Basically you have change factors for all 365 days from the equations in the paper. First we apply a smoothing function to them because we don’t want neighbouring days to change by a lot more than each other. Then you just apply the CHANGE for a day to each hour in that day in the weather file.

 

[[TQM]] For example say that in the weather file January 1st had 10 mm precip in the 1st hour of the day, 20 mm in the 2nd, and 30 mm in the 3rd then no other rainfall in the day at all, and the scaling factor for the day (computed from the climate projections) is 1.10 or 10%. Then the new weather file would have 11 mm, 22 mm, and 33 mm, respectively in those 3 hours.

 

I think its possible I may be misunderstanding what is actually being generated by the run, as some of the closing portions of this make little sense to me. The actual processing of the data doesn't seem like it will be that terrible once I've gotten my hands of the netCDF files i need, but I don't think I could describe what is being generated (leading me to believe I don't know what is being generated).

 

[[TQM]] No worries, I think you’re right – the hard part is wrapping your head around exactly what is happening and once that’s clear the programming will be fairly straightforward. Again, I’m happy to answer these questions – feel free to keep them coming if still unclear.

------------------------------------------------------------------------------------------------------------------------------ 

First and foremost, what are the angled brackets around a variable meant in this paper? I believe you mean them to be the mean value for the day d of the variable x, however I'm not confident on this. 

[[TQM]] The angled brackets indicate an average over day in the 30-year period for the given day d, where d=1…365 (Jan 1 through Dec 31) and the 30 year period is either 2011-2040, 2041-2070, or 2071-2100 for future and always 1971-2000 for past. Does that make sense?

 

Next, when calculating both deltas and alphas, are two different years worth of known data used? for example would each file input have 2 distinct years in them, or would each run expect 2 inputs, with each file having its own year? otherwise what is xfuture vs xpresent?

 

[[TQM]] Xfuture and Xpresent are referring to time in the time series from the NetCDF file for the climate model. The temporal resolution is daily, starting in 1951 and going to 2100. So there are actually 150 years present there. But you are making two 30 year averages first so you have an average for each day of the year for both the past and that future. The deltas and alphas are computed based on that.

 

[[TQM]] The weather file only has a single year in it and it’s temporal “resolution” is hourly.

 

Finally, would you be able to send some netCDF files, and their expected outputs so that I can test what I write is correct? Hopefully those questions make sense, I believe I understand the rest of what is going on, but if I come across anything else while I'm doing this I'll let you know.


[[TQM]] Yes, the questions make sense and mostly reflect the fact that you haven’t seen one of the NetCDF files yet. I’m still in the midst of a slew of events for the next 2 days so I’m copying Steve Sobie – he’ll be able to provide a sample NetCDF file that you can use. (Steve, I don’t want to just point Ross at the data portal because I want him to use the same NetCDF files we will be access later directly at PCIC). I will have email access though so if you have any more questions please feel free to ask.


	
