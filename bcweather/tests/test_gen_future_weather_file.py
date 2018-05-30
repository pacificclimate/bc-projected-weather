import io

import numpy

from bcweather import get_epw_header, get_climate_data


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
    print(data)
    print(numpy.array(data).shape)
    assert data.any()
