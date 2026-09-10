import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_v7_external_population import select


def pair(name,image_hash,reference_hash):
    return {"files":[{"path":f"camera/{name}.tiff","sha256":image_hash},{"path":f"camera/{name}.wp","sha256":reference_hash}]}


def test_population_excludes_shared_reference_without_claiming_a_new_scene():
    old=[pair("a","image0","reference0")]
    new=[pair("b","image1","reference0"),pair("c","image2","reference2")]
    rows=select(new,old)
    assert [r["primary"] for r in rows]==[False,True]
    assert all(r["role"]=="external_test_only" for r in rows)
    with pytest.raises(ValueError,match="Historical image overlap"):
        select([pair("b","image0","reference2")],old)
    with pytest.raises(ValueError,match="Duplicate"):
        select(new+new,old)
