"""Known-source TRAIN routing test of frozen patch color experts."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
import torch
from skin_capture_model import CaptureColor,MODES
from skin_capture_support import make_plan,apply_plan
from skin_capture_support_train import OUT as SOURCE,RUN as SOURCE_RUN
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import write
from skin_mskcc_audit import scalar_de
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_capture_support_v1'
RUN=ROOT/'experiments/runs/skin_capture_support_v1'
PROTOCOL=ROOT/'docs/research/skin_local_routing_probe_v1.md'


def route(hypotheses,weights,gate):
    h=np.asarray(hypotheses,dtype=float);w=np.asarray(weights,dtype=float);g=np.asarray(gate,dtype=float)
    if g.ndim==2:return ((h*w[:,:,None,None]).sum(1)*g[:,:,None]).sum(1)
    return ((h*g[:,:,:,None]).sum(2)*w[:,:,None]).sum(1)


def main():
    records=[SOURCE/f'mixed__{arm}__s{s}'/'result.json' for arm in ('baseline','paired_union') for s in (17,29,43)]
    files=[Path(__file__),PROTOCOL,ROOT/'tests/test_skin_local_routing_probe.py',ROOT/'scripts/skin_capture_model.py',
        ROOT/'scripts/skin_capture_support.py',ROOT/'scripts/skin_mskcc_audit.py',ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz']+records
    files += [SOURCE_RUN/p.parent.name/'best.pt' for p in records]
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'routing_probe_lock.json'
    if not lock.exists():write(lock,{'bindings':bindings});print('Routing probe frozen; checkpoint before next invocation');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    t=load('train');pairs=np.array([p for site in np.unique(t['site']) for p in itertools.combinations(np.flatnonzero(t['site']==site),2)])
    assert len(pairs)==1421
    mode=np.eye(4,dtype=np.float32)[[MODES.index(str(m)) for m in t['mode']]]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    result=[];gap=0.;cases=0;input_hashes=[]
    for path in records:
        r=json.loads(path.read_bytes());f=SOURCE_RUN/path.parent.name/'best.pt';assert sha(f)==r['best_sha256']
        state=torch.load(f,map_location='cpu',weights_only=True);model=CaptureColor('mixture').cuda();model.load_state_dict(state['state']);model.eval()
        mean=state['target_mean'].cuda();std=state['target_std'].cuda()
        rng=np.random.default_rng(4017);prng=np.random.default_rng(9123);digest=hashlib.sha256();outputs={k:[] for k in ('learned_global','known_global','known_weighted_global','known_local','shuffled_local')}
        with torch.no_grad():
            for start in range(0,len(pairs),32):
                pp=pairs[start:start+32];n=len(pp);ix=np.r_[pp[:,0],pp[:,1]];partner=np.r_[n:2*n,0:n]
                np.testing.assert_array_equal(t['target'][pp[:,0]],t['target'][pp[:,1]])
                x=torch.from_numpy(t['tokens'][ix]).cuda();m=torch.from_numpy(mode[ix]).cuda();plan=make_plan(2*n,64,rng);plan['augment'][:]=True
                inp,_,origin,_=apply_plan(x,m,torch.from_numpy(partner).cuda(),plan,'paired_union')
                local=torch.where(origin[...,None],m[torch.from_numpy(partner).cuda(),None,:],m[:,None,:])[:n]
                inp=inp[:n];digest.update(inp.cpu().numpy().tobytes());digest.update(local.cpu().numpy().tobytes())
                h=model.local(inp);ctx=model.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
                v=model.votes(torch.cat([h,ctx[:,None].expand(-1,64,-1)],-1))
                weights=v[...,12].softmax(1).cpu().numpy()
                hypothesis=(v[...,:12].reshape(n,64,4,3)*std+mean).cpu().numpy()
                gate=model.gate(ctx).softmax(1).cpu().numpy();local=local.cpu().numpy()
                shuffled=np.stack([a[prng.permutation(64)] for a in local])
                known=local.mean(1);weighted=(local*weights[...,None]).sum(1)
                for name,g in [('learned_global',gate),('known_global',known),('known_weighted_global',weighted),('known_local',local),('shuffled_local',shuffled)]:outputs[name].append(route(hypothesis,weights,g))
        outputs={k:np.concatenate(v) for k,v in outputs.items()};target=t['target'][pairs[:,0]];scores={}
        for name,pred in outputs.items():
            error=delta_e00(pred,target);ind=np.array([scalar_de(a,b) for a,b in zip(pred,target)]);cases+=len(ind)
            gap=max(gap,float(np.max(abs(ind-error))))
            scores[name]={'mean':float(ind.mean()),'median':float(np.median(ind)),'p95':float(np.quantile(ind,.95))}
        file=RUN/path.parent.name/'routing_probe.npz';np.savez(file,**outputs,target=target)
        result.append({'source_model':path.parent.name,'scores':scores,'input_sha256':digest.hexdigest(),'artifact_sha256':sha(file)})
        input_hashes.append(digest.hexdigest());print(json.dumps(result[-1]),flush=True)
        del model;torch.cuda.empty_cache()
    assert gap<1e-9 and len(set(input_hashes))==1
    write(OUT/'routing_probe.json',{'scope':'Teacher-assisted TRAIN virtual-bag diagnostic; no model improvement claim',
        'pairs':len(pairs),'records':result,'independent_scalar_cases':cases,'maximum_gap':gap,
        'identical_inputs_and_permutations_for_all_models':True,'no_routing_uses_native_Lab':True,
        'reserved_endpoint_access':False,'bindings':bindings})


if __name__=='__main__':main()
