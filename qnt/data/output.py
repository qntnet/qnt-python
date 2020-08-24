from qnt.data.common import *
import numpy as np
import pandas as pd
import xarray as xr
import gzip
import qnt.data.id_translation as idt


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
    output.coords[ds.ASSET] = [idt.translate_user_id_to_server_id(id) for id in output.coords[ds.ASSET].values]
    output = sort_and_crop_output(output)
    data = output.to_netcdf(compute=True)
    data = gzip.compress(data)
    path = get_env("OUTPUT_PATH", "fractions.nc.gz")
    print("write output: " + path)
    with open(path, 'wb') as out:
        out.write(data)