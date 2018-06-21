import io

import pytest
import numpy
import pandas

from bcweather import get_epw_header, get_climate_data, morph_data
from bcweather import get_daily_averages, gen_prism_offset_weather_file
from bcweather.epw import epw_to_data_frame


def test_get_epw_header():
    my_string = """Line 1
Line 2
More stuff
Other stuff
Still more worthless data
What else could *possibly* in this file?!
Line 7
Line 8
Line 9
"""
    f = io.StringIO(my_string)
    pos = 15
    f.seek(pos)
    rv = get_epw_header(f)
    assert rv.startswith("Line 1")
    assert rv.endswith("Line 8\n")
    assert len(rv.splitlines()) == 8
    assert f.tell() == pos


def test_get_climate_data(ncfile):
    data = get_climate_data(ncfile, 50.8, -118.38, 'tasmax', [1970, 1972])
    ##print(data)
    print(numpy.array(data).shape)
    assert data.any()


def test_epw_to_data_frame(epwfile):
    df = epw_to_data_frame(epwfile)
    assert type(df) == pandas.DataFrame
    assert 'datetime' in df.columns
    assert 'dry_bulb_temperature' in df.columns


def test_morph_data():
    with pytest.raises(NotImplementedError):
        morph_data(0, 0, 0, 0, 0)

##def test_get_daily_averages(epwfile):
##    df = epw_to_data_frame(epwfile)
##    x = get_daily_averages(df['dry_bulb_temperature'], df['datetime'])
##    assert len(x) == 365

def test_gen_prism_offset_weather_file():
    epw_header = gen_prism_offset_weather_file(49.2,-123.2)
    print(epw_header)
    assert type(epw_header) == str
