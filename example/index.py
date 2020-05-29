import qnt.data as qndata
import time
import datetime as dt

idx_list = qndata.load_index_list( tail=dt.timedelta(days=365))

print(idx_list)

idx_data = qndata.load_index_data(forward_order=False, tail=dt.timedelta(days=365))

print(idx_data)
