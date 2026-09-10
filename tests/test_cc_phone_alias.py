import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_phone_alias import open_camera_rgb


@pytest.mark.parametrize("key",["samsung","MIS"])
def test_single_rgb_key_is_read_without_transform(tmp_path,key):
    x = np.arange(18,dtype=np.float32).reshape(2,3,3)
    path = tmp_path/"samsung.h5"
    with h5py.File(path,"w") as f:
        f[key] = x
    with open_camera_rgb(path,"samsung") as actual:
        np.testing.assert_array_equal(actual[:],x)


@pytest.mark.parametrize("second,channels",[(True,3),(False,16)])
def test_ambiguous_or_multispectral_input_is_rejected(tmp_path,second,channels):
    path = tmp_path/"samsung.h5"
    with h5py.File(path,"w") as f:
        f["MIS"] = np.ones((2,3,channels),dtype=np.float32)
        if second:
            f["samsung"] = np.ones((2,3,3),dtype=np.float32)
    with pytest.raises(ValueError):
        with open_camera_rgb(path,"samsung"):
            pass
