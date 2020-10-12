from qnt.data.common import *


def load_major_index_list():
    uri = "major-idx/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)
    return idx


def load_major_index_data(
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
):
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "major-idx/data?min_date=" + str(min_date) + "&max_date=" + str(max_date)
    raw = request_with_retry(uri, None)

    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})
    arr.name = "major_indexes"
    return arr.transpose(*dims)


def load_index_list(
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
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

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

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
        assets: tp.Union[None, tp.List[tp.Union[str,dict]]] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str] = (ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL,
) -> tp.Union[None, xr.DataArray]:
    max_date = parse_date(max_date)

    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    if assets is not None:
        assets = [a['id'] if type(a) == dict else a for a in assets]

    if assets is None:
        assets_array = load_index_list(min_date, max_date)
        assets_arg = [i['id'] for i in assets_array]
    else:
        assets_arg = assets

    params = {"ids": assets_arg, "min_date": min_date.isoformat(), "max_date": max_date.isoformat()}
    params = json.dumps(params)
    params = params.encode()
    raw = request_with_retry("idx/data", params)

    if raw is None or len(raw) < 1:
        return None

    arr = xr.open_dataarray(raw, cache=True, decode_times=True)
    arr = arr.compute()

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    if assets is not None:
        assets = sorted(assets)
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        arr = arr.broadcast_like(assets)

    arr.name = "indexes"
    return arr.transpose(*dims)