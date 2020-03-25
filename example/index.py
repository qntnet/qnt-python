import qnt.data as qndata
import time

idx_list = qndata.load_index_list()

print(idx_list)

idx_data = qndata.load_index_data(forward_order=False)

print(idx_data)
