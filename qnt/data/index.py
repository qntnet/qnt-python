from qnt.data.common import *


def load_index_list(
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, None] = None
) -> list:
    """
    :return: list of dicts with info for all indexes
    """
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if tail is None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - tail

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
        assets: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date] = '2007-01-01',
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str] = (ds.TIME, ds.ASSET),
        forward_order: bool = False,
        tail: tp.Union[datetime.timedelta, None] = None
) -> tp.Union[None, xr.DataArray]:
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if tail is None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - tail

    if assets is None:
        assets = load_index_list(min_date, max_date)
        assets = [i['id'] for i in assets]
    params = {"ids": assets, "min_date": min_date.isoformat(), "max_date": max_date.isoformat()}
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