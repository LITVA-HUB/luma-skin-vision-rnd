import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_offset_diagnostic import excluded_person_offsets


def test_own_reference_cannot_change_own_correction():
    patient=np.array(['a','a','b','c']);site=np.array(['x','x','y','z'])
    pred=np.zeros((4,3));target=np.array([[1,2,3],[1,2,3],[4,5,6],[8,9,10]],float)
    a=excluded_person_offsets(pred,target,patient,site)
    changed=target.copy();changed[patient=='a']+=1000
    b=excluded_person_offsets(pred,changed,patient,site)
    np.testing.assert_array_equal(a[patient=='a'],b[patient=='a'])
    np.testing.assert_allclose(a[patient=='a'],[[6,7,8],[6,7,8]])


def test_photo_replication_does_not_reweight_sites_or_people():
    patient=np.array(['a','b','b','c']);site=np.array(['x','y','z','w'])
    pred=np.zeros((4,3));target=np.array([[0,0,0],[2,2,2],[6,6,6],[10,10,10]],float)
    a=excluded_person_offsets(pred,target,patient,site)
    ix=np.array([0,1,1,1,2,3])
    b=excluded_person_offsets(pred[ix],target[ix],patient[ix],site[ix])
    np.testing.assert_allclose(a[0],[7,7,7]);np.testing.assert_array_equal(a[0],b[0])
