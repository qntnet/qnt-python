from qnt.data.common import *
import xarray as  xr
import json


def load_futures_list() -> xr.DataArray:
    uri = "futures/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_futures_data(
        assets: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        max_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
) -> tp.Union[None, xr.DataArray]:
    if max_date is None and "LAST_DATA_PATH" in os.environ:
        whole_data_file_flag_name = get_env("LAST_DATA_PATH", "last_data.txt")
        with open(whole_data_file_flag_name, "w") as text_file:
            text_file.write("last")

    max_date = parse_date_and_hour(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date is not None:
        min_date = parse_date_and_hour(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "futures/data"
    raw = request_with_retry(uri, None)
    if raw is None or len(raw) < 1:
        return None

    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()

    arr = arr.sel(time=slice(max_date.date().isoformat(), min_date.date().isoformat(), 1))

    if assets is not None:
        assets = [a['id'] if type(a) == dict else a for a in assets]

    if assets is not None:
        arr = arr.sel(asset=assets)
        assets = sorted(assets)
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        arr = arr.broadcast_like(assets)

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    arr.name = "futures"
    return arr.transpose(*dims)
