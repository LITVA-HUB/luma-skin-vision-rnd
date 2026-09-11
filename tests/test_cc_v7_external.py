import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_v7_external_data import publisher_pair
from cc_v7_external_report import family_measure, paired_draws


def test_publisher_rgb_whitepoint_alignment_and_saturation(tmp_path):
    bgr=np.empty((8,8,3),dtype=np.uint16)
    bgr[:]=[10000,20000,30000]
    bgr[0,0]=65535
    ok,encoded=cv2.imencode(".tiff",bgr)
    assert ok
    payloads={"image":encoded.tobytes(),"reference":b"0.2 0.3 0.4\n"}
    row={}
    for kind,content in payloads.items():
        name=kind+(".tiff" if kind=="image" else ".wp")
        (tmp_path/name).write_bytes(content)
        row[kind]={"path":name,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
    rgb,gt,shape=publisher_pair(tmp_path,row)
    np.testing.assert_allclose(rgb[1,1],np.array([30000,20000,10000])/65535,rtol=1e-6)
    np.testing.assert_array_equal(rgb[0,0],np.zeros(3))
    np.testing.assert_allclose(gt,np.array([.2,.3,.4])/np.linalg.norm([.2,.3,.4]))
    assert shape==[8,8,3]
    row["image"]["path"]="../escape.tiff"
    with pytest.raises(ValueError,match="escaped"):
        publisher_pair(tmp_path,row)


def test_proxy_bootstrap_never_splits_a_reference_group():
    rows=[{"group":"a","camera":"A"},{"group":"a","camera":"A"},{"group":"b","camera":"A"},{"group":"c","camera":"B"}]
    a=paired_draws(rows,repeats=20,seed=17)
    b=paired_draws(rows,repeats=20,seed=17)
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x,y)
        assert np.sum(x==0)==np.sum(x==1)
        assert np.sum(x==3)==1
    rows[1]["camera"]="B"
    with pytest.raises(ValueError,match="spans cameras"):
        paired_draws(rows,repeats=1)


def test_bootstrap_fixed_coverage_counts_rejections():
    data={"error":np.array([[1,2,3,4,5]]),"score":np.array([[5,4,3,2,1]]),"valid":np.ones((1,5),dtype=bool)}
    indices=np.arange(5)
    keys=np.array([str(i) for i in indices])
    assert family_measure(data,indices,keys)==(3.,3.5)
    data["valid"][0,:2]=False
    assert family_measure(data,indices,keys)==(3.,None)
