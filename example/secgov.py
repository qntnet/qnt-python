import qnt.data as qndata
import time

i = 0
j = 0
st = time.time()
for f in qndata.load_secgov_forms(
        types=['10-Q'],
        # facts=[
        #     'us-gaap:EarningsPerShareDiluted',
        #     'us-gaap:Liabilities',
        #     'us-gaap:Assets',
        #     'us-gaap:CommonStockSharesOutstanding'
        # ],
        # skip_segment=True,
        min_date='2017-01-01'
):
    # print(f['url'], len(f['facts']))
    print(i, j, f['date'], time.time() - st)
    i += 1
    j += len(f['facts'])
