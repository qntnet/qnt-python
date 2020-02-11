from .data import f, ds, load_assets, sort_and_crop_output, get_env
import xarray as xr
import numpy as np
import bottleneck
import pandas as pd
import gzip, base64, json
from urllib import parse, request
from tabulate import tabulate

EPS = 10 ** -7


def calc_slippage(data, period=14, fract=0.05):
    """
    :param data: xarray with historical data
    :param period: lookback period
    :param fract: slippage factor
    :return: xarray with slippage
    """
    time_series = np.sort(data.coords[ds.TIME])
    data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET).loc[[f.CLOSE, f.HIGH, f.LOW], time_series, :]

    cl = data.loc[f.CLOSE].shift({ds.TIME: 1})
    hi = data.loc[f.HIGH]
    lo = data.loc[f.LOW]
    d1 = hi - lo
    d2 = abs(hi - cl)
    d3 = abs(cl - lo)
    dd = xr.concat([d1, d2, d3], dim='d').max(dim='d', skipna=False)
    dd = dd.rolling({ds.TIME: period}, min_periods=period).mean(skipna=False).ffill(ds.TIME)
    return dd * fract


def calc_relative_return(data, portfolio_history, slippage_factor=0.05, per_asset=False):
    target_weights = portfolio_history.shift(**{ds.TIME: 1})[1:]  # shift and cut first point

    slippage = calc_slippage(data, 14, slippage_factor)

    data, target_weights, slippage = arrange_data(data, target_weights, slippage, per_asset)

    W = target_weights
    D = data

    OPEN = D.loc[f.OPEN].ffill(ds.TIME).fillna(0)
    CLOSE = D.loc[f.CLOSE].ffill(ds.TIME).fillna(0)
    DIVS = D.loc[f.DIVS].fillna(0)

    # boolean matrix when assets available for trading
    UNLOCKED = np.logical_and(np.isfinite(D.loc[f.OPEN].values), np.isfinite(D.loc[f.CLOSE].values))
    UNLOCKED = np.logical_and(np.isfinite(W.values), UNLOCKED)
    UNLOCKED = np.logical_and(np.isfinite(slippage.values), UNLOCKED)
    UNLOCKED = np.logical_and(OPEN > EPS, UNLOCKED)

    if per_asset:
        RR = W.copy(True)
        RR[:] = calc_relative_return_np_per_asset(W.values, UNLOCKED, OPEN.values, CLOSE.values, slippage.values,
                                                  DIVS.values)
        return RR
    else:
        RR = xr.DataArray(
            np.full([len(W.coords[ds.TIME])], np.nan, np.double),
            dims=[ds.TIME],
            coords={ds.TIME: W.coords[ds.TIME]}
        )
        RR[:] = calc_relative_return_np(W.values, UNLOCKED, OPEN.values, CLOSE.values, slippage.values, DIVS.values)
        return RR


def calc_relative_return_np_per_asset(WEIGHT, UNLOCKED, OPEN, CLOSE, SLIPPAGE, DIVS):
    N = np.zeros(WEIGHT.shape)  # shares count

    equity_before_buy = np.zeros(WEIGHT.shape)
    equity_after_buy = np.zeros(WEIGHT.shape)
    equity_tonight = np.zeros(WEIGHT.shape)

    for t in range(0, WEIGHT.shape[0]):
        unlocked = UNLOCKED[t]  # available for trading

        if t == 0:
            equity_before_buy[0] = 1
            N[0] = 0
        else:
            N[t] = N[t - 1]
            equity_before_buy[t] = equity_after_buy[t - 1] + (OPEN[t] - OPEN[t - 1] + DIVS[t]) * N[t]

        N[t, unlocked] = equity_before_buy[t, unlocked] * WEIGHT[t, unlocked] / OPEN[t, unlocked]
        dN = N[t]
        if t > 0:
            dN = dN - N[t - 1]
        S = SLIPPAGE[t] * abs(dN)  # slippage for this step
        equity_after_buy[t] = equity_before_buy[t] - S
        equity_tonight[t] = equity_after_buy[t] + (CLOSE[t] - OPEN[t]) * N[t]

        locked = np.logical_not(unlocked)
        if t == 0:
            equity_before_buy[0, locked] = 1
            equity_after_buy[0, locked] = 1
            equity_tonight[0, locked] = 1
            N[0, locked] = 0
        else:
            N[t, locked] = N[t - 1, locked]
            equity_after_buy[t, locked] = equity_after_buy[t - 1, locked]
            equity_before_buy[t, locked] = equity_before_buy[t - 1, locked]
            equity_tonight[t, locked] = equity_tonight[t - 1, locked]

    E = equity_tonight
    Ep = np.roll(E, 1, axis=0)
    Ep[0] = 1
    RR = E / Ep - 1
    RR = np.where(np.isfinite(RR), RR, 0)
    return RR


def calc_relative_return_np(WEIGHT, UNLOCKED, OPEN, CLOSE, SLIPPAGE, DIVS):
    N = np.zeros(WEIGHT.shape)  # shares count

    equity_before_buy = np.zeros([WEIGHT.shape[0]])
    equity_operable_before_buy = np.zeros([WEIGHT.shape[0]])
    equity_after_buy = np.zeros([WEIGHT.shape[0]])
    equity_tonight = np.zeros([WEIGHT.shape[0]])

    for t in range(0, WEIGHT.shape[0]):
        unlocked = UNLOCKED[t]  # available for trading
        locked = np.logical_not(unlocked)

        if t == 0:
            equity_before_buy[0] = 1
            N[0] = 0
        else:
            N[t] = N[t - 1]
            equity_before_buy[t] = equity_after_buy[t - 1] + np.nansum((OPEN[t] - OPEN[t - 1] + DIVS[t]) * N[t])

        w_sum = np.nansum(abs(WEIGHT[t]))
        w_free_cash = max(1, w_sum) - w_sum
        w_unlocked = np.nansum(abs(WEIGHT[t, unlocked]))
        w_operable = w_unlocked + w_free_cash

        equity_operable_before_buy[t] = equity_before_buy[t] - np.nansum(OPEN[t, locked] * abs(N[t, locked]))

        if w_operable < EPS:
            equity_after_buy[t] = equity_before_buy[t]
        else:
            N[t, unlocked] = equity_operable_before_buy[t] * WEIGHT[t, unlocked] / (w_operable * OPEN[t, unlocked])
            dN = N[t, unlocked]
            if t > 0:
                dN = dN - N[t - 1, unlocked]
            S = np.nansum(SLIPPAGE[t, unlocked] * abs(dN))  # slippage for this step
            equity_after_buy[t] = equity_before_buy[t] - S

        equity_tonight[t] = equity_after_buy[t] + np.nansum((CLOSE[t] - OPEN[t]) * N[t])

    E = equity_tonight
    Ep = np.roll(E, 1)
    Ep[0] = 1
    RR = E / Ep - 1
    RR = np.where(np.isfinite(RR), RR, 0)
    return RR


def arrange_data(data, target_weights, additional_series=None, per_asset=False):
    """
    arranges data for proper calculations
    :param per_asset:
    :param data:
    :param target_weights:
    :param additional_series:
    :return:
    """
    min_date = target_weights.coords[ds.TIME].min().values
    max_date = data.coords[ds.TIME].max().values

    if additional_series is not None:
        additional_series_without_nan = additional_series.dropna(ds.TIME, 'all')
        min_date = max(min_date, additional_series_without_nan.coords[ds.TIME].min().values)
        max_date = min(max_date, additional_series_without_nan.coords[ds.TIME].max().values)

    time_series = data.coords[ds.TIME]

    time_series = time_series.where(np.logical_and(time_series >= min_date, time_series <= max_date)).dropna(ds.TIME)
    time_series.values = np.sort(time_series)

    assets = np.intersect1d(target_weights.coords[ds.ASSET].values, data.coords[ds.ASSET].values, True)
    assets = np.sort(assets)

    adjusted_data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    adjusted_data = adjusted_data.loc[:, time_series, assets]

    adjusted_tw = xr.DataArray(
        np.full([len(time_series), len(assets)], np.nan, dtype=np.float64),
        dims=[ds.TIME, ds.ASSET],
        coords={
            ds.TIME: time_series,
            ds.ASSET: assets
        }
    )

    time_intersected = np.intersect1d(time_series.values, target_weights.coords[ds.TIME].values, True)

    weights_intersection = target_weights.transpose(ds.TIME, ds.ASSET).loc[time_intersected, assets]
    weights_intersection = weights_intersection.where(np.isfinite(weights_intersection)).fillna(0)

    adjusted_tw.loc[time_intersected, assets] = weights_intersection
    adjusted_tw = adjusted_tw.where(
        np.logical_or(np.logical_not(np.isnan(adjusted_tw.values)), adjusted_data.loc[f.IS_LIQUID] > 0), 0)

    if per_asset:
        adjusted_tw = xr.where(adjusted_tw > 1, 1, adjusted_tw)
        adjusted_tw = xr.where(adjusted_tw < -1, -1, adjusted_tw)
    else:
        s = abs(adjusted_tw).sum(ds.ASSET)
        s = xr.where(s < 1, 1, s)
        adjusted_tw = adjusted_tw / s

    if additional_series is not None:
        additional_series = additional_series.loc[time_series]
        if ds.ASSET in additional_series.dims:
            additional_series = additional_series.loc[:, assets]

    try:
        adjusted_tw = adjusted_tw.drop(ds.FIELD)
    except ValueError:
        pass

    return (adjusted_data, adjusted_tw, additional_series)


def calc_equity(relative_return):
    """
    :param relative_return: daily return
    :return: daily portfolio equity
    """
    return (relative_return + 1).cumprod(ds.TIME)


def calc_volatility(relative_return, max_periods=252, min_periods=2):
    """
    :param relative_return: daily return
    :param max_periods: maximal number of days
    :param min_periods: minimal number of days
    :return: portfolio volatility
    """
    max_periods = min(max_periods, len(relative_return.coords[ds.TIME]))
    min_periods = min(min_periods, max_periods)
    return relative_return.rolling({ds.TIME: max_periods}, min_periods=min_periods).std()


def calc_volatility_annualized(relative_return, max_periods=252, min_periods=2):
    """
    :param relative_return: daily return
    :param min_periods: minimal number of days
    :return: annualized volatility
    """
    return calc_volatility(relative_return, max_periods, min_periods) * pow(252, 1. / 2)


def calc_underwater(equity):
    """
    :param equity: daily portfolio equity
    :return: daily underwater
    """
    mx = equity.rolling({ds.TIME: len(equity)}, min_periods=1).max()
    return equity / mx - 1


def calc_max_drawdown(underwater):
    """
    :param underwater: daily underwater
    :return: daily maximum drawdown
    """
    return (underwater).rolling({ds.TIME: len(underwater)}, min_periods=1).min()


def calc_sharpe_ratio_annualized(relative_return, max_periods=252, min_periods=2):
    """
    :param relative_return: daily return
    :param max_periods: maximal number of days
    :param min_periods: minimal number of days
    :return: annualized Sharpe ratio
    """
    m = calc_mean_return_annualized(relative_return, max_periods, min_periods)
    v = calc_volatility_annualized(relative_return, max_periods, min_periods)
    sr = m / v
    return sr


def calc_mean_return(relative_return, max_periods=252, min_periods=1):
    """
    :param relative_return: daily return
    :param max_periods: maximal number of days
    :param min_periods: minimal number of days
    :return: daily mean return
    """
    max_periods = min(max_periods, len(relative_return.coords[ds.TIME]))
    min_periods = min(min_periods, max_periods)
    return xr.ufuncs.exp(
        xr.ufuncs.log(relative_return + 1).rolling({ds.TIME: max_periods}, min_periods=min_periods).mean(
            skipna=True)) - 1


def calc_mean_return_annualized(relative_returns, max_periods=252, min_periods=1):
    """
    :param relative_returns: daily return
    :param min_periods: minimal number of days
    :return: annualized mean return
    """
    power = func_np_to_xr(np.power)
    return power(calc_mean_return(relative_returns, max_periods, min_periods) + 1, 252) - 1


def calc_bias(portfolio_history, per_asset=False):
    """
    :param per_asset:
    :param portfolio_history: portfolio weights set for every day
    :return: daily portfolio bias
    """
    if per_asset:
        return portfolio_history
    ph = portfolio_history
    sum = ph.sum(ds.ASSET)
    abs_sum = abs(ph).sum(ds.ASSET)
    res = sum / abs_sum
    res = res.where(np.isfinite(res)).fillna(0)
    return res


def calc_instruments(portfolio_history, per_asset=False):
    """
    :param per_asset:
    :param portfolio_history: portfolio weights set for every day
    :return: daily portfolio instrument count
    """
    if per_asset:
        I = portfolio_history.copy(True)
        I[:] = 1
        return I
    ph = portfolio_history.copy().fillna(0)
    ic = ph.where(ph == 0).fillna(1)
    ic = ic.cumsum(ds.TIME)
    ic = ic.where(ic == 0).fillna(1)
    ic = ic.sum(ds.ASSET)
    return ic


def calc_avg_turnover(portfolio_history, equity, data, max_periods=252, min_periods=1, per_asset=False):
    '''
    Calculates average capital turnover, all args must be adjusted
    :param portfolio_history: history of portfolio changes
    :param equity: equity of changes
    :param data:
    :param max_periods:
    :param min_periods:
    :param per_asset:
    :return:
    '''
    W = portfolio_history.transpose(ds.TIME, ds.ASSET)
    W = W.shift({ds.TIME: 1})
    W[0] = 0

    Wp = W.shift({ds.TIME: 1})
    Wp[0] = 0

    OPEN = data.transpose(ds.TIME, ds.FIELD, ds.ASSET).loc[W.coords[ds.TIME], f.OPEN, W.coords[ds.ASSET]]
    OPENp = OPEN.shift({ds.TIME: 1})
    OPENp[0] = OPEN[0]

    E = equity

    Ep = E.shift({ds.TIME: 1})
    Ep[0] = 1

    turnover = abs(W - Wp * Ep * OPEN / (OPENp * E))
    if not per_asset:
        turnover = turnover.sum(ds.ASSET)
    max_periods = min(max_periods, len(turnover.coords[ds.TIME]))
    min_periods = min(min_periods, len(turnover.coords[ds.TIME]))
    turnover = turnover.rolling({ds.TIME: max_periods}, min_periods=min_periods).mean()
    try:
        turnover = turnover.drop(ds.FIELD)
    except ValueError:
        pass
    return turnover


def calc_non_liquid(data, portfolio_history):
    (adj_data, adj_ph, ignored) = arrange_data(data, portfolio_history, None)
    if f.IS_LIQUID in list(adj_data.coords[ds.FIELD]):
        non_liquid = adj_ph.where(
            np.logical_and(np.isfinite(adj_data.loc[f.IS_LIQUID]), adj_data.loc[f.IS_LIQUID] == 0))
        non_liquid = non_liquid.dropna(ds.ASSET, 'all')
        non_liquid = non_liquid.dropna(ds.TIME, 'all')
        if abs(non_liquid).sum() > 0:
            return non_liquid
    return None


def find_missed_dates(output, data):
    out_ts = np.sort(output.coords[ds.TIME].values)

    min_out_ts = min(out_ts)

    data_ts = data.coords[ds.TIME]
    data_ts = data_ts.where(data_ts >= min_out_ts).dropna(ds.TIME)
    data_ts = np.sort(data_ts.values)
    return np.array(np.setdiff1d(data_ts, out_ts))


def func_np_to_xr(origin_func):
    '''
    Decorates numpy function for xarray
    '''
    func = xr.ufuncs._UFuncDispatcher(origin_func.__name__)
    func.__name__ = origin_func.__name__
    doc = origin_func.__doc__
    func.__doc__ = ('xarray specific variant of numpy.%s. Handles '
                    'xarray.Dataset, xarray.DataArray, xarray.Variable, '
                    'numpy.ndarray and dask.array.Array objects with '
                    'automatic dispatching.\n\n'
                    'Documentation from numpy:\n\n%s' % (origin_func.__name__, doc))
    return func


class StatFields:
    RELATIVE_RETURN = "relative_return"
    EQUITY = "equity"
    VOLATILITY = "volatility"
    UNDERWATER = "underwater"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    MEAN_RETURN = "mean_return"
    BIAS = "bias"
    INSTRUMENTS = "instruments"
    AVG_TURNOVER = "avg_turnover"


stf = StatFields


def calc_stat(data, portfolio_history, slippage_factor=0.05, min_periods=2,
              max_periods=252 * 3, per_asset=False):
    """
    :param data: xarray with historical data, data must be split adjusted
    :param portfolio_history: portfolio weights set for every day
    :param slippage_factor:
    :param min_periods: minimal number of days
    :param max_periods: max number of days for rolling
    :param per_asset: calculate stats per asset
    :return: xarray with all statistics
    """
    portfolio_history = sort_and_crop_output(portfolio_history, per_asset)
    non_liquid = calc_non_liquid(data, portfolio_history)
    if non_liquid is not None:
        print("WARNING: Strategy trades non-liquid assets.")

    missed_dates = find_missed_dates(portfolio_history, data)
    if len(missed_dates) > 0:
        print("WARNING: some dates are missed in the portfolio_history")

    RR = calc_relative_return(data, portfolio_history, slippage_factor, per_asset)

    E = calc_equity(RR)
    V = calc_volatility_annualized(RR, max_periods=max_periods, min_periods=min_periods)
    U = calc_underwater(E)
    DD = calc_max_drawdown(U)
    SR = calc_sharpe_ratio_annualized(RR, max_periods=max_periods, min_periods=min_periods)
    MR = calc_mean_return_annualized(RR, max_periods=max_periods, min_periods=min_periods)
    (adj_data, adj_ph, ignored) = arrange_data(data, portfolio_history, E, per_asset)
    B = calc_bias(adj_ph, per_asset)
    I = calc_instruments(adj_ph, per_asset)
    T = calc_avg_turnover(adj_ph, E, adj_data, min_periods=min_periods, max_periods=max_periods, per_asset=per_asset)

    stat = xr.concat([
        E, RR, V,
        U, DD, SR,
        MR, B, I, T
    ], pd.Index([
        stf.EQUITY, stf.RELATIVE_RETURN, stf.VOLATILITY,
        stf.UNDERWATER, stf.MAX_DRAWDOWN, stf.SHARPE_RATIO,
        stf.MEAN_RETURN, stf.BIAS, stf.INSTRUMENTS, stf.AVG_TURNOVER
    ], name=ds.FIELD))

    dims = [ds.TIME, ds.FIELD]
    if per_asset:
        dims.append(ds.ASSET)
    return stat.transpose(*dims)


def calc_sector_distribution(portfolio_history, timeseries=None):
    """
    :param portfolio_history: portfolio weights set for every day
    :param timeseries: time range
    :return: sector distribution
    """
    ph = abs(portfolio_history.transpose(ds.TIME, ds.ASSET)).fillna(0)
    s = ph.sum(ds.ASSET)
    s[s < 1] = 1
    ph = ph / s

    if timeseries is not None:  # arrange portfolio to timeseries
        _ph = xr.DataArray(np.full([len(timeseries), len(ph.coords[ds.ASSET])], 0, dtype=np.float64),
                           dims=[ds.TIME, ds.ASSET],
                           coords={
                               ds.TIME: timeseries,
                               ds.ASSET: ph.coords[ds.ASSET]
                           })
        intersection = np.intersect1d(timeseries, ph.coords[ds.TIME], True)
        _ph.loc[intersection] = ph.loc[intersection]
        ph = _ph.ffill(ds.TIME).fillna(0)

    max_date = str(portfolio_history.coords[ds.TIME].max().values)[0:10]
    min_date = str(portfolio_history.coords[ds.TIME].min().values)[0:10]

    assets = load_assets(min_date=min_date, max_date=max_date)
    assets = dict((a['id'], a) for a in assets)

    sectors = []

    SECTOR_FIELD = 'sector'

    for aid in portfolio_history.coords[ds.ASSET].values:
        sector = "Other"
        if aid in assets:
            asset = assets[aid]
            s = asset[SECTOR_FIELD]
            if s is not None and s != 'n/a' and s != '':
                sector = s
        sectors.append(sector)

    uq_sectors = sorted(list(set(sectors)))
    sectors = np.array(sectors)

    CASH_SECTOR = 'Cash'
    sector_distr = xr.DataArray(
        np.full([len(ph.coords[ds.TIME]), len(uq_sectors) + 1], 0, dtype=np.float64),
        dims=[ds.TIME, SECTOR_FIELD],
        coords={
            ds.TIME: ph.coords[ds.TIME],
            SECTOR_FIELD: uq_sectors + [CASH_SECTOR]
        }
    )

    for sector in uq_sectors:
        sum_by_sector = ph.loc[:, sectors == sector].sum(ds.ASSET)
        sector_distr.loc[:, sector] = sum_by_sector

    sector_distr.loc[:, CASH_SECTOR] = 1 - ph.sum(ds.ASSET)

    return sector_distr


def print_correlation(portfolio_history, data):
    """ Checks correlation for current output. """
    portfolio_history = sort_and_crop_output(portfolio_history)
    rr = calc_relative_return(data, portfolio_history)

    cr_list = calc_correlation(rr)

    print()

    if len(cr_list) == 0:
        print("Ok. This strategy does not correlate with other strategies.")
        return

    print("WARNING! This strategy correlates with other strategies.")
    print("The number of systems with a larger Sharpe ratio and correlation larger than 0.8:", len(cr_list))
    print("The max correlation value (with systems with a larger Sharpe ratio):", max([i['cofactor'] for i in cr_list]))
    my_cr = [i for i in cr_list if i['my']]

    print("Current sharpe ratio(3y):", calc_sharpe_ratio_annualized(rr, 253 * 3)[-1].values.item())

    print()

    if len(my_cr) > 0:
        print("My correlated submissions:\n")
        headers = ['Name', "Coefficient", "Sharpe ratio"]
        rows = []

        for i in my_cr:
            rows.append([i['name'], i['cofactor'], i['sharpe_ratio']])

        print(tabulate(rows, headers))


def calc_correlation(relative_returns):
    ENGINE_CORRELATION_URL = get_env("ENGINE_CORRELATION_URL",
                                     "http://localhost:8080/referee/submission/forCorrelation")
    STATAN_CORRELATION_URL = get_env("STATAN_CORRELATION_URL", "http://localhost:8081/statan/correlation")
    PARTICIPANT_ID = get_env("PARTICIPANT_ID", "0")

    with request.urlopen(ENGINE_CORRELATION_URL + "?participantId=" + PARTICIPANT_ID) as response:
        submissions = response.read()
        submissions = json.loads(submissions)
        submission_ids = [s['id'] for s in submissions]

    rr = relative_returns.to_netcdf(compute=True)
    rr = gzip.compress(rr)
    rr = base64.b64encode(rr)
    rr = rr.decode()

    r = {"relative_returns": rr, "submission_ids": submission_ids}
    r = json.dumps(r)
    r = r.encode()

    with request.urlopen(STATAN_CORRELATION_URL, r) as response:
        cofactors = response.read()
        cofactors = json.loads(cofactors)

    result = []
    for c in cofactors:
        sub = next((s for s in submissions if str(c['id']) == str(s['id'])))
        sub['cofactor'] = c['cofactor']
        sub['sharpe_ratio'] = c['sharpe_ratio']
        result.append(sub)

    return result
