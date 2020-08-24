import typing as tp
import datetime
import os
import logging
from urllib.parse import urljoin
import sys
import urllib.request
import time
import json
import math
import xarray as xr
import numpy as np


MAX_DATE_LIMIT: tp.Union[datetime.date, None] = None


class Fields:
    OPEN = "open"
    LOW = "low"
    HIGH = "high"
    CLOSE = "close"
    VOL = "vol"
    DIVS = "divs"
    SPLIT = "split"
    SPLIT_CUMPROD = "split_cumprod"
    IS_LIQUID = "is_liquid"
    ROLL = "roll"


f = Fields


class Dimensions:
    TIME = 'time'
    FIELD = 'field'
    ASSET = 'asset'


ds = Dimensions

TIMEOUT = 60
RETRY_DELAY = 1


def get_env(key, def_val):
    if key in os.environ:
        return os.environ[key]
    else:
        print("WARNING: env is not set " + key)
        return def_val


BASE_URL = get_env('DATA_BASE_URL', 'http://127.0.0.1:8000/')


def request_with_retry(uri, data):
    url = urljoin(BASE_URL, uri)
    retries = sys.maxsize if "SUBMISSION_ID" in os.environ else 5
    for r in range(0, retries):
        try:
            with urllib.request.urlopen(url, data, timeout=TIMEOUT) as response:
                response_body = response.read()
                return response_body
        except KeyboardInterrupt:
            raise
        except:
            logging.exception("download error " + uri)
            time.sleep(RETRY_DELAY)
    raise Exception("can't download " + uri)


def parse_date(dt: tp.Union[None, str, datetime.datetime, datetime.date]) -> datetime.date:
    if dt is None:
        try:
            return parse_date_and_hour(dt).date()
        except:
            return datetime.date.today()
    if isinstance(dt, str):
        return datetime.datetime.strptime(dt + "Z+00:00", "%Y-%m-%dZ%z").date()
    if isinstance(dt, datetime.datetime):
        dt = datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc)  # rm timezone
        return dt.date()
    if isinstance(dt, datetime.date):
        return dt
    raise Exception("invalid date")



def parse_date_and_hour(dt: tp.Union[None, str, datetime.datetime, datetime.date]) -> datetime.datetime:
    if dt is None:
        try:
            dt = BASE_URL.split("/")[-2]
            return parse_date_and_hour(dt)
        except:
            return datetime.datetime.now(tz=datetime.timezone.utc)
    if isinstance(dt, datetime.date):
        return datetime.datetime(dt.year, dt.month, dt.day, tzinfo=datetime.timezone.utc)
    if isinstance(dt, datetime.datetime):
        dt = datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc)  # rm timezone
        dt = dt.isoformat()
    if isinstance(dt, str):
        dt = dt.split(":")[0]
        if 'T' in dt:
            return datetime.datetime.strptime(dt + "Z+00:00", "%Y-%m-%dT%HZ%z")
        else:
            return datetime.datetime.strptime(dt + "Z+00:00", "%Y-%m-%dZ%z")
    raise Exception("invalid date")


def datetime_to_hours_str(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H")


# TODO Strange stuff, need to check usage

def from_xarray_3d_to_dict_of_pandas_df(xarray_data):
    assets_names = xarray_data.coords[ds.ASSET].values
    pandas_df_dict = {}
    for asset_name in assets_names:
        pandas_df_dict[asset_name] = xarray_data.loc[:, :, asset_name].to_pandas()

    return pandas_df_dict


def from_dict_to_xarray_1d(weights):
    weights_assets_list = [key for key in weights]
    weights_values_list = [weights[key] for key in weights]

    return xr.DataArray(weights_values_list, dims=[ds.ASSET], coords={ds.ASSET: weights_assets_list})


def filter_liquids_xarray_assets_dataarray(assets_xarray_dataarray):
    liquid_xarray_assets_dataarray = assets_xarray_dataarray \
        .where(assets_xarray_dataarray.loc[:, 'is_liquid', :] == 1) \
        .dropna(ds.TIME, 'all').dropna(ds.ASSET, 'all')

    return liquid_xarray_assets_dataarray


def check_weights_xarray_dataarray_for_nonliquids(xarray_weights_dataarray, xarray_assets_dataarray):
    non_liquid_weights = xarray_weights_dataarray.where(xarray_assets_dataarray[0].loc['is_liquid', :] == 0)
    non_liquid_weights = non_liquid_weights.where(non_liquid_weights != 0)
    non_liquid_weights = non_liquid_weights.dropna(ds.ASSET)
    if len(non_liquid_weights) > 0:
        raise Exception(non_liquid_weights.coords[ds.ASSET].values)


def exclude_weights_xarray_dataarray_from_nonliquids(weights_xarray_dataarray, assets_xarray_dataarray):
    liquid_weights_xarray_dataarray = weights_xarray_dataarray \
        .where(assets_xarray_dataarray[0].loc['is_liquid', :] == 1) \
        .dropna(ds.ASSET, 'all')

    return liquid_weights_xarray_dataarray


