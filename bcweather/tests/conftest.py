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
        'tests/data/CAN_BC_Abbotsford.Intl.AP.711080_CWEC2016.epw'
    )
    return(fname)
 
@pytest.fixture
def prismfiles():
    pr_name = resource_filename(
        'bcweather',
        'tests/data/small_pr_monClim_PRISM_historical_run1_198101-201012.nc'
    )
    tx_name = resource_filename(
        'bcweather',
        'tests/data/small_tmax_monClim_PRISM_historical_run1_198101-201012.nc'
    )
    tn_name = resource_filename(
        'bcweather',
        'tests/data/small_tmin_monClim_PRISM_historical_run1_198101-201012.nc'
    )
    prism_files = [pr_name , tx_name, tn_name]
    return prism_files

@pytest.fixture
def alphatas():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_tasmax_tasmin_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_tasmax_tasmin_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_tasmax_tasmin_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    alphatas = [acc_name , can_name, cnr_name]
    return alphatas

@pytest.fixture
def deltatas():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_delta_tas_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_delta_tas_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_delta_tas_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    deltatas = [acc_name , can_name, cnr_name]
    return deltatas

@pytest.fixture
def alphadew():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_dewpoint_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_dewpoint_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_dewpoint_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    alphadew = [acc_name , can_name, cnr_name]
    return alphadew

@pytest.fixture
def deltadew():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_delta_dewpoint_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_delta_dewpoint_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_delta_dewpoint_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    deltadew = [acc_name , can_name, cnr_name]
    return deltadew

@pytest.fixture
def alpharad():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_rsds_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_rsds_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_rsds_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    alpharsds = [acc_name , can_name, cnr_name]
    return alpharsds

@pytest.fixture
def alphaclt():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_clt_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_clt_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_clt_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    alphaclt = [acc_name , can_name, cnr_name]
    return alphaclt

@pytest.fixture
def alpharhs():
    acc_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_rhs_ACCESS1-0_1971-2000_2041-2070.nc'
    )
    can_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_rhs_CanESM2_1971-2000_2041-2070.nc'
    )
    cnr_name = resource_filename(
        'bcweather',
        'tests/data/small_alpha_rhs_CNRM-CM5_1971-2000_2041-2070.nc'
    )
    alpharhs = [acc_name , can_name, cnr_name]
    return alpharhs



