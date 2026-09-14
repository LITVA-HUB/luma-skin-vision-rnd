"""Independent NumPy check of a fixed snapshot of completed HR inner banks."""
import gc

import numpy as np
import torch
from chromaseed_architecture_scale import base_model
from chromaseed_gated import unpack
from chromaseed_head_range import Predictor
from chromaseed_head_range_run import RUN, TIMES, load_data
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context
from skin_local_search_train import sha, write_json
from threadpoolctl import threadpool_limits


def main():
    assert not torch.cuda.is_initialized(), 'This checker must remain CPU-only'
    lock = js(RUN/'source_lock.json')
    for name in ['scripts/chromaseed_head_range.py', 'scripts/chromaseed_head_range_fit.py',
                 'scripts/chromaseed_head_range_run.py', 'scripts/chromaseed_architecture_scale.py']:
        assert sha(ROOT/name) == lock['sources'][name]
    paths = sorted((RUN/'inner').glob('*/*/fold*/receipt.json'))
    assert paths, 'No complete inner banks yet'
    snapshot = {p.relative_to(RUN).as_posix():sha(p) for p in paths}
    dest = RUN/'inner_audits'/f'completed_{len(paths):04d}.json'
    protocol = dict(source_sha256=sha(__file__), source_lock_sha256=sha(RUN/'source_lock.json'),
                    bank_receipts=snapshot, atol_native_lab=.002, rtol=1e-6,
                    scope='all saved checkpoint/slot OOF vectors in these completed inner banks; no new held-role inference')
    protocol_path = dest.with_name(dest.stem+'_protocol.json')
    if protocol_path.exists():
        assert js(protocol_path) == protocol
    else:
        write_json(protocol_path,protocol)
    if dest.exists():
        result=js(dest)
        assert result['protocol_sha256']==sha(protocol_path)
        print('Existing completed snapshot preserved',sha(dest),flush=True)
        return
    data = load_data()
    records=[]
    with threadpool_limits(limits=1):
        for receipt_path in paths:
            path=receipt_path.parent
            receipt=js(receipt_path)
            assert receipt['source_lock_sha256']==sha(RUN/'source_lock.json')
            assert receipt['selection_sha256'] is None
            for name,digest in receipt['files'].items():
                assert sha(path/name)==digest
            ix,query,warm,parents=context(data,receipt['role'],receipt['fold'])
            assert receipt['warm_parent_sha256']==parents
            rows=nz(path/'rows.npz')
            np.testing.assert_array_equal(rows['fit_rows'],ix)
            np.testing.assert_array_equal(rows['query_rows'],query)
            assert not set(data['patient'][ix]) & set(data['patient'][query])
            tokens=data['tokens'][ix].astype(float)
            token_mean=tokens.mean((0,1)).astype(np.float32)
            token_std=np.maximum(tokens.std((0,1)),1e-6).astype(np.float32)
            maximum=0.
            models_checked,vectors=0,0
            for step in TIMES:
                bank=nz(path/f'models_{step}.npz')
                saved=nz(path/f'oof_{step}.npz')
                np.testing.assert_array_equal(saved['row_indices'],query)
                assert saved['predictions'].shape==(6,len(query),3)
                for slot in range(6):
                    model=unpack(bank,str(slot))
                    assert str(model['head_mode'])==receipt['head_mode']
                    np.testing.assert_array_equal(model['t_mean'],token_mean)
                    np.testing.assert_array_equal(model['t_std'],token_std)
                    b=base_model(model)
                    for key,value in warm[slot//2].items():
                        np.testing.assert_array_equal(b[key],value)
                    consumer=Predictor(model)
                    pred=np.concatenate([consumer(data['color'][q],data['tokens'][q]) for q in np.array_split(query,max(1,(len(query)+15)//16))])
                    expected=saved['predictions'][slot]
                    np.testing.assert_allclose(pred,expected,atol=.002,rtol=1e-6)
                    maximum=max(maximum,float(np.abs(pred-expected).max()))
                    models_checked+=1
                    vectors+=len(query)
                del bank,saved,consumer,model
                gc.collect()
            record=dict(bank=path.relative_to(RUN).as_posix(),models=models_checked,query_vectors=vectors,
                        maximum_lab_difference=maximum,fit_people=len(set(data['patient'][ix])),query_people=len(set(data['patient'][query])))
            records.append(record)
            print('HR INNER NUMPY VERIFIED',record['bank'],models_checked,vectors,maximum,flush=True)
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,protocol_sha256=sha(protocol_path),banks=len(records),
                models=sum(r['models'] for r in records),query_vectors=sum(r['query_vectors'] for r in records),
                maximum_lab_difference=max(r['maximum_lab_difference'] for r in records),
                cuda_context_initialized=False,records=records,not_a_full_primary_audit=True)
    write_json(dest,result)
    print('HR INNER CHECK PASSED',sha(dest),flush=True)


if __name__=='__main__':
    main()
