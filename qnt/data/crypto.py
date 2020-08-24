from qnt.data.common import *

def load_cryptocurrency_data(
        assets: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date, datetime.datetime] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = False,
        tail: tp.Union[datetime.timedelta, None] = None
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

    if tail is None:
        min_date = parse_date_and_hour(min_date)
    else:
        min_date = max_date - tail

    uri = "crypto?min_date=" + datetime_to_hours_str(min_date) + "&max_date=" + datetime_to_hours_str(max_date)
    raw = request_with_retry(uri, None)
    if raw is None or len(raw) < 1:
        return None

    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()

    if assets is not None:
        arr = arr.sel(asset=assets)

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    return arr.transpose(*dims)


BATCH_LIMIT = 300000
