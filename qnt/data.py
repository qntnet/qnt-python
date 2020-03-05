import xarray as xr
import numpy as np
import time
import bottleneck
import urllib.request
import json
import scipy
import datetime
import math
import sys
import os
import gzip
from urllib.parse import urljoin
import logging
from qnt.id_translation import *
import qnt.id_translation as idt
import typing as tp

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
    IS_LIQUID = 'is_liquid'


f = Fields


class Dimensions:
    TIME = 'time'
    FIELD = 'field'
    ASSET = 'asset'


ds = Dimensions

TIMEOUT = 60

def load_assets(
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None
):
    """
    :return: list of dicts with info for all tickers
    """
    min_date = parse_date(min_date)
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date > max_date:
        raise Exception("min_date must be less than or equal to max_date")

    # print(str(max_date))

    uri = "assets?min_date=" + str(min_date) + "&max_date=" + str(max_date)
    js = request_with_retry(uri, None)
    js = js.decode()
    tickers = json.loads(js)

    tickers.sort(key=lambda a: str(a.get('last_point', '0000-00-00')) + "_" + a['id'], reverse=True)
    for t in tickers:
        t['id'] = translate_asset_to_user_id(t)
        t.pop('last_point', None)
    tickers.sort(key=lambda a: a['id'])

    return tickers


def load_data(
        assets: tp.List[str] = None,
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = False
) -> xr.DataArray:
    """
    :param assets: list of ticker names to load
    :param min_date: first date in data
    :param max_date: last date of data
    :param dims: tuple with ds.FIELD, ds.TIME, ds.ASSET in the specified order
    :param forward_order: boolean, set true if you need the forward order of dates, otherwise the order is backward
    :return: xarray DataArray with historical data for selected assets
    """
    if assets is None:
        assets_array = load_assets(min_date=min_date, max_date=max_date)
        assets = [a['id'] for a in assets_array]
    t = time.time()
    data = load_origin_data(assets=assets, min_date=min_date, max_date=max_date)
    print("Data loaded " + str(round(time.time() - t)) + "s")
    if data is None:
        return None
    data = adjust_by_splits(data, False)
    data = data.transpose(*dims)
    if forward_order:
        data = data.sel(**{ds.TIME: slice(None, None, -1)})
    return data


def sort_and_crop_output(output, per_asset=False):
    output = output.where(np.isfinite(output)).where(output != 0).dropna(ds.ASSET, 'all').fillna(0)
    output = output.transpose(ds.TIME, ds.ASSET)
    output = output.loc[np.sort(output.coords[ds.TIME].values), np.sort(output.coords[ds.ASSET].values)]
    if per_asset:
        output = xr.where(output > 1, 1, output)
        output = xr.where(output < -1, -1, output)
    else:
        s = abs(output).sum(ds.ASSET)
        s[s < 1] = 1
        output = output / s
    try:
        output = output.drop(ds.FIELD)
    except ValueError:
        pass
    return output


def write_output(output):
    """
    writes output in the file for submission
    :param output: xarray with daily weights
    """
    output = output.copy()
    output.coords[ds.ASSET] = [translate_user_id_to_server_id(id) for id in output.coords[ds.ASSET].values]
    output = sort_and_crop_output(output)
    data = output.to_netcdf(compute=True)
    data = gzip.compress(data)
    path = get_env("OUTPUT_PATH", "fractions.nc.gz")
    print("write output: " + path)
    with open(path, 'wb') as out:
        out.write(data)


def adjust_by_splits(data, make_copy=True):
    """
    :param data: xarray
    :param make_copy: if True the initial data isn't changed
    :return: xarray with data adjusted by splits
    """
    if make_copy:
        data = data.copy()
    dims = data.dims
    data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    data.loc[f.OPEN] = data.loc[f.OPEN] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.LOW] = data.loc[f.LOW] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.HIGH] = data.loc[f.HIGH] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.CLOSE] = data.loc[f.CLOSE] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.VOL] = data.loc[f.VOL] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.DIVS] = data.loc[f.DIVS] * data.loc[f.SPLIT_CUMPROD]
    return data.transpose(*dims)


def restore_origin_data(data, make_copy=True):
    """
    :param data: xarray
    :param make_copy: if True the initial data isn't changed
    :return: xarray with origin data
    """
    if make_copy:
        data = data.copy()
    dims = data.dims
    data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    data.loc[f.OPEN] = data.loc[f.OPEN] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.LOW] = data.loc[f.LOW] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.HIGH] = data.loc[f.HIGH] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.CLOSE] = data.loc[f.CLOSE] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.VOL] = data.loc[f.VOL] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.DIVS] = data.loc[f.DIVS] / data.loc[f.SPLIT_CUMPROD]
    return data.transpose(*dims)


def load_secgov_forms(
        ciks: tp.Union[None, tp.List[str]] = None,
        types: tp.Union[None, tp.List[str]] = None,
        facts: tp.Union[None, tp.List[str]] = None,
        skip_segment: bool = False,
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None
) -> tp.Generator[dict, None, None]:
    """
    Load SEC Forms (Fundamental data)
    :param ciks: list of cik (you can get cik from asset id)
    :param types: list of form types: ['10-K', '10-Q', '10-K/A', '10-Q/A']
    :param facts: list of facts for extraction, for example: ['us-gaap:Goodwill']
    :param skip_segment: skip facts with segment
    :param min_date: min form date
    :param max_date: max form date
    :return:
    """
    params = {
        'ciks': ciks,
        'types': types,
        'facts': facts,
        'skip_segment': skip_segment,
        'min_date': parse_date(min_date).isoformat(),
        'max_date': parse_date(max_date).isoformat()
    }
    go = True
    while go:
        params_js = json.dumps(params)
        raw = request_with_retry("sec.gov/forms", params_js.encode())
        js = raw.decode()
        forms = json.loads(js)
        for f in forms:
            yield f
        go = len(forms) > 0
        params['offset'] = params.get('offset', 0) + len(forms)


def load_index_list(
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None
) -> list:
    """
    :return: list of dicts with info for all indexes
    """
    min_date = parse_date(min_date)
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date > max_date:
        raise Exception("min_date must be less than or equal to max_date")

    # print(str(max_date))

    uri = "idx/list?min_date=" + str(min_date) + "&max_date=" + str(max_date)
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)

    idx.sort(key=lambda a: a['id'])

    return idx


def load_index_data(
        ids: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str] = (ds.TIME, ds.ASSET),
        forward_order: bool = False
) -> tp.Union[None, xr.DataArray]:
    if ids is None:
        ids = load_index_list(min_date, max_date)
        ids = [i['id'] for i in ids]
    params = {"ids": ids, "min_date": min_date, "max_date": max_date}
    params = json.dumps(params)
    params = params.encode()
    raw = request_with_retry("idx/data", params)

    if raw is None or len(raw) < 1:
        return None

    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})
    return arr.transpose(*dims)


def load_cryptocurrency_data(
        min_date: tp.Union[str, datetime.date, datetime.datetime] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = False
) -> tp.Union[None, xr.DataArray]:
    if max_date is None:
        max_date = datetime_to_hours_str(datetime.datetime.now(tz=datetime.timezone.utc))
    min_date = parse_date_and_hour(min_date)
    max_date = parse_date_and_hour(max_date)
    uri = "crypto?min_date=" + datetime_to_hours_str(min_date) + "&max_date=" + datetime_to_hours_str(max_date)
    raw = request_with_retry(uri, None)
    if raw is None or len(raw) < 1:
        return None

    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    return arr.transpose(*dims)


BATCH_LIMIT = 300000

def load_origin_data(assets, min_date, max_date=None):
    assets = [translate_user_id_to_server_id(id) for id in assets]
    # load data from server
    if max_date is None and "LAST_DATA_PATH" in os.environ:
        whole_data_file_flag_name = get_env("LAST_DATA_PATH", "last_data.txt")
        with open(whole_data_file_flag_name, "w") as text_file:
            text_file.write("last")

    min_date = parse_date(min_date)
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    # print(str(max_date))

    if min_date > max_date:
        raise Exception("min_date must be less than or equal to max_date")

    start_time = time.time()

    days = (max_date - min_date).days + 1
    chunk_asset_count = math.floor(BATCH_LIMIT / days)

    chunks = []

    for offset in range(0, len(assets), chunk_asset_count):
        chunk_assets = assets[offset:(offset + chunk_asset_count)]
        chunk = load_origin_data_chunk(chunk_assets, min_date.isoformat(), max_date.isoformat())
        if chunk is not None:
            chunks.append(chunk)
        print(
            "fetched chunk "
            + str(round(offset / chunk_asset_count + 1)) + "/"
            + str(math.ceil(len(assets) / chunk_asset_count)) + " "
            + str(round(time.time() - start_time)) + "s"
        )
    if len(chunks) == 0:
        return None

    whole = xr.concat(chunks, ds.ASSET)
    whole = whole.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    whole.coords[ds.ASSET] = [translate_server_id_to_user_id(id) for id in whole.coords[ds.ASSET].values]
    whole = whole.loc[
        [f.OPEN, f.LOW, f.HIGH, f.CLOSE, f.VOL, f.DIVS, f.SPLIT, f.SPLIT_CUMPROD, f.IS_LIQUID],
        np.sort(whole.coords[ds.TIME])[::-1],
        np.sort(whole.coords[ds.ASSET])
    ]

    return whole


def load_origin_data_chunk(assets, min_date, max_date):  # min_date and max_date - iso date str
    params = {
        'assets': assets,
        'min_date': min_date,
        'max_date': max_date
    }
    params = json.dumps(params)
    raw = request_with_retry("data", params.encode())
    if len(raw) == 0:
        return None
    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()
    return arr


def get_env(key, def_val):
    if key in os.environ:
        return os.environ[key]
    else:
        print("WARNING: env is not set " + key)
        return def_val


BASE_URL = get_env('DATA_BASE_URL', 'http://127.0.0.1:8000/')
RETRY_DELAY = 1


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
        return datetime.datetime.now(tz=datetime.timezone.utc)
    if isinstance(dt, datetime.date):
        return datetime.datetime(dt.year, dt.month, dt.day, datetime.timezone.utc)
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


if idt.USE_ID_TRANSLATION is None:
    js = request_with_retry('assets', None)
    js = js.decode()
    tickers = json.loads(js)
    idt.USE_ID_TRANSLATION = next((i for i in tickers if i.get('FIGI') is not None), None) is not None

if __name__ == '__main__':
    # import qnt.id_translation
    # qnt.id_translation.USE_ID_TRANSLATION = False
    assets = load_assets()
    print(len(assets))
    ids = [i['id'] for i in assets]
    print(ids)
    data = load_data(min_date='1998-11-09', assets=ids[-2000:])
    print(data.sel(field='close').transpose().to_pandas())


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
