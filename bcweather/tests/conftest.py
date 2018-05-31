from pkg_resources import resource_filename

import pytest
import netCDF4


@pytest.fixture
def ncfile():
    fname = resource_filename('bcweather', 'tests/data/tiny_downscaled.nc')
    dst = netCDF4.Dataset(fname, 'r')
    return dst


@pytest.fixture
def epwfile():
    fname = resource_filename(
        'bcweather',
        'tests/data/CAN_BC_ABBOTSFORD-A_1100031_CWEC.epw'
    )
    with open(fname, 'r') as f:
        yield f
