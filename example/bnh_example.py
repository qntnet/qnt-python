import qnt.data as qndata
import qnt.stats as qnstats
import qnt.forward_looking as qnfl
import time
from qnt.neutralization import neutralize
import datetime as dt
import qnt.exposure as qne

assets = qndata.load_assets()

data = qndata.load_data(
    assets=[a['id'] for a in assets[-15:]],
    max_date='2020-03-01',
    tail=dt.timedelta(days=365),
    forward_order=True,
    dims=("time", "field", "asset"))

data_b = qndata.load_data(
    # assets=[a['id'] for a in assets[:2000]],
    max_date='2020-03-01',
    tail=dt.timedelta(days=365),
    forward_order=True,
    dims=("time", "field", "asset"))

print(qnstats.calc_avg_points_per_year(data))

output = data.sel(field=qndata.f.IS_LIQUID)
output = qndata.sort_and_crop_output(output)
output = neutralize(output, assets, 'industry')

output *= 1

print(output.to_pandas())
print(output[0, 0].item())

#print(qnstats.calc_slippage(data).to_pandas()[-100:])

# stat2 = qnstats.calc_stat(data, output, slippage_factor=0.05, per_asset=True)
# # ss = qnstats.calc_stat(data, output, max_periods=252 * 3, slippage_factor=0.05, per_asset=True)
#
# print(stat2.sel(field=[  # qnstats.stf.RELATIVE_RETURN,
#     qnstats.stf.MEAN_RETURN,
#     # qnstats.stf.VOLATILITY,
#     qnstats.stf.SHARPE_RATIO,
#     # qnstats.stf.EQUITY,
#     # qnstats.stf.MAX_DRAWDOWN
#     qnstats.stf.AVG_HOLDINGTIME
# ]).isel(asset=0).to_pandas())

#output = qne.drop_bad_days(output)

out2 = data_b.sel(field='is_liquid')

qnstats.check_exposure(output)

print("mix")

mix = qne.mix_weights(output, data_b.sel(field='is_liquid'))

qnstats.check_exposure(mix)


