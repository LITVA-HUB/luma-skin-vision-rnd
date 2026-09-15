"""Fixed pretraining and person-held-out transfer; never selects using OOF."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from model import Model, lab_head


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def fresh(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()): raise RuntimeError(f'Nonempty output preserved: {p}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(path):
    m = Model()
    c = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(c['model'], strict=True)
    return m, c


def freeze_check(args):
    if not args.freeze: raise ValueError('Training requires a preregistered freeze')
    d = json.loads(Path(args.freeze).read_text())
    for p, expected in d['files'].items():
        if sha(p) != expected: raise ValueError(f'Frozen file changed: {p}')
    if d['config_sha256'] != sha(args.config): raise ValueError('Configuration changed')
    return sha(args.freeze)


def cache_read(path):
    p = Path(path);r = json.loads((p/'receipt.json').read_text())
    for n, s in r['outputs'].items():
        if sha(p/n) != s: raise ValueError('Cache changed')
    if sha(r['metadata']) != r['metadata_sha256']: raise ValueError('Metadata changed')
    return torch.from_numpy(np.load(p/'early.npy')), torch.from_numpy(np.load(p/'mask.npy')), dict(np.load(r['metadata'], allow_pickle=False)), r


def init(args, c):
    p = fresh(args.out);torch.manual_seed(c['seed'])
    m = Model(c['pretrained'])
    torch.save({'model':m.state_dict(), 'config_sha256':sha(args.config)}, p/'initial.pt')
    write(p/'receipt.json', {'sha256':sha(p/'initial.pt'), 'parameters':sum(x.numel() for x in m.parameters()), 'stem':state_sha(m.encoder.stem.state_dict())})


def cache(args,c):
    p = fresh(args.out);m,_ = load_model(args.initial);m.eval()
    prefix = Path(c['data'])/args.source
    rgb = np.load(str(prefix)+'_rgb.npy', mmap_mode='r')
    assert rgb.dtype == np.uint8 and rgb.shape[1:] == (3,128,128)
    masks = np.load(str(prefix)+'_mask.npy', mmap_mode='r') if args.source == 'stw' else None
    e = np.empty((len(rgb),48,8,8), np.float32);z = np.ones((len(rgb),1,4,4),np.float32)
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode():
        for i in range(0,len(rgb),64):
            x=torch.from_numpy(np.array(rgb[i:i+64])).float()/255
            e[i:i+64]=m.encoder.stem((x-mean)/std).numpy()
            if masks is not None:
                mk=torch.from_numpy(np.array(masks[i:i+64])).float()
                assert set(torch.unique(mk).tolist()) <= {0.,1.}
                z[i:i+64]=nn.functional.adaptive_avg_pool2d(mk,(4,4)).numpy()
    np.save(p/'early.npy',e);np.save(p/'mask.npy',z)
    meta=str(prefix)+'_meta.npz'
    write(p/'receipt.json',{'source':args.source,'rows':len(rgb),'metadata':meta,'metadata_sha256':sha(meta),'image_sha256':sha(str(prefix)+'_rgb.npy'),'initial_sha256':sha(args.initial),'stem':state_sha(m.encoder.stem.state_dict()),'outputs':{n:sha(p/n) for n in ['early.npy','mask.npy']}})


def log(p,d):
    with (p/'training.jsonl').open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n')
    print(json.dumps(d),flush=True)


def optim(params,lr,c):
    return torch.optim.AdamW(params,lr=lr,weight_decay=c['weight_decay'],betas=(.9,.999),eps=1e-8,foreach=False)


def pretrain(args,c):
    fsha=freeze_check(args);p=fresh(args.out);e,mk,meta,receipt=cache_read(args.cache)
    torch.manual_seed(c['seed']);m,initial=load_model(args.initial);m.train()
    assert receipt['stem']==state_sha(m.encoder.stem.state_dict())
    cfg=c['pretrain'];o=optim([x for x in m.parameters() if x.requires_grad],cfg['lr'],cfg)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(o,cfg['epochs'],eta_min=cfg['eta_min'])
    idx=np.flatnonzero(meta['split']=='train');assert len(idx)==9406
    # Source-scoped group keys prevent accidental pairing of equal IDs in different sources.
    keys=np.array([f'{s}|{g}' for s,g in zip(meta['source'],meta['group'])])
    pools={key:idx[keys[idx]==key] for key in np.unique(keys[idx])}
    for key,ix in pools.items():
        assert len(np.unique(meta['mst'][ix]))==1
    labels=torch.tensor(meta['mst']).long()
    assert labels.min()>=1 and labels.max()<=10
    generator=torch.Generator().manual_seed(c['seed']);start=0
    initial_stem=state_sha(m.encoder.stem.state_dict());initial_tail=state_sha(m.encoder.tail.state_dict())
    execution={'variant':args.variant,'config_sha256':sha(args.config),'freeze_sha256':fsha,'initial_sha256':sha(args.initial),'cache_receipt_sha256':sha(Path(args.cache)/'receipt.json')}
    if args.resume:
        z=torch.load(args.resume,map_location='cpu',weights_only=False);assert z['execution']==execution
        m.load_state_dict(z['model']);o.load_state_dict(z['optimizer']);scheduler.load_state_dict(z['scheduler']);start=z['epoch'];generator.set_state(z['generator']);torch.set_rng_state(z['rng'])
    write(p/'execution.json',execution)
    for epoch in range(start,cfg['epochs']):
        ts=time.perf_counter();order=idx[torch.randperm(len(idx),generator=generator).numpy()];losses=[];pairs=0;lr=o.param_groups[0]['lr']
        for pos in range(0,len(order),cfg['anchors']):
            a=order[pos:pos+cfg['anchors']];b=[]
            for i in a:
                candidates=pools[keys[i]];candidates=candidates[candidates!=i]
                b.append(int(candidates[torch.randint(len(candidates),(1,),generator=generator).item()]) if len(candidates) else int(i))
            b=np.asarray(b);both=np.concatenate([a,b]);pair=torch.tensor(a!=b);pairs+=int(pair.sum())
            logits,emb=m(e[both],mk[both]);targets=(labels[both,None]>torch.arange(1,10)[None]).float()
            ordinal=nn.functional.binary_cross_entropy_with_logits(logits,targets)
            consistency=(1-nn.functional.cosine_similarity(emb[:len(a)][pair],emb[len(a):][pair])).mean() if pair.any() else emb.sum()*0
            loss=ordinal+cfg['consistency'][args.variant]*consistency
            assert torch.isfinite(loss)
            o.zero_grad(set_to_none=True);loss.backward()
            assert all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())
            nn.utils.clip_grad_norm_(m.parameters(),cfg['clip'],error_if_nonfinite=True);o.step()
            losses.append([float(ordinal.detach()),float(consistency.detach()),len(both)])
        scheduler.step();values=np.array(losses);means=np.average(values[:,:2],weights=values[:,2],axis=0)
        log(p,{'epoch':epoch+1,'lr':lr,'ordinal':means[0],'consistency':means[1],'real_repeat_pairs':pairs,'seconds':time.perf_counter()-ts})
        if epoch+1 in [6,cfg['epochs']]:
            torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'generator':generator.get_state(),'rng':torch.get_rng_state(),'execution':execution},p/f'epoch_{epoch+1:03d}.pt')
    torch.save({'model':m.state_dict(),'execution':execution,'epoch':cfg['epochs']},p/'weights.pt')
    write(p/'verification.json',{'stem_unchanged':initial_stem==state_sha(m.encoder.stem.state_dict()),'tail_changed':initial_tail!=state_sha(m.encoder.tail.state_dict()),'weights_sha256':sha(p/'weights.pt'),'epochs':cfg['epochs'],'heldout_used_for_fit':False})


def transfer(args,c):
    fsha=freeze_check(args);p=fresh(args.out);e,mk,meta,receipt=cache_read(args.cache)
    torch.manual_seed(c['seed']+args.fold);model,_=load_model(args.initial);encoder=model.encoder
    assert receipt['stem']==state_sha(encoder.stem.state_dict())
    cfg=c['transfer'];train=np.flatnonzero(meta['fold']!=args.fold);hold=np.flatnonzero(meta['fold']==args.fold)
    assert len(meta['image'])==966 and len(np.unique(meta['patient']))==24
    assert len(np.unique(meta['patient'][hold]))==4
    assert not set(meta['patient'][hold])&set(meta['patient'][train])
    target=torch.tensor(meta['target'],dtype=torch.float32);assert meta['target'].dtype==np.float64
    encoder.eval()
    with torch.inference_mode():emb=torch.cat([encoder(e[i:i+128],mk[i:i+128]) for i in range(0,len(e),128)])
    norm={'xm':emb[train].mean(0),'xs':emb[train].std(0,unbiased=False).clamp_min(1e-6),'ym':target[train].mean(0),'ys':target[train].std(0,unbiased=False).clamp_min(1e-6)}
    head=lab_head();limited=args.mode=='limited';encoder.requires_grad_(limited);encoder.stem.requires_grad_(False)
    params=[{'params':head.parameters(),'lr':cfg['head_lr']}]
    if limited:params.append({'params':[v for v in encoder.parameters() if v.requires_grad],'lr':cfg['encoder_lr']})
    o=optim(params,cfg['head_lr'],cfg);g=torch.Generator().manual_seed(c['seed']+args.fold);start=0
    original_stem=state_sha(encoder.stem.state_dict());original_encoder=state_sha(encoder.state_dict())
    execution={'variant':args.variant,'mode':args.mode,'fold':args.fold,'config_sha256':sha(args.config),'freeze_sha256':fsha,'initial_sha256':sha(args.initial),'cache_receipt_sha256':sha(Path(args.cache)/'receipt.json')}
    epochs=cfg[args.mode+'_epochs']
    if args.resume:
        z=torch.load(args.resume,map_location='cpu',weights_only=False);assert z['execution']==execution
        head.load_state_dict(z['head']);encoder.load_state_dict(z['encoder_delta'],strict=False);o.load_state_dict(z['optimizer']);norm=z['norm'];start=z['epoch'];g.set_state(z['generator']);torch.set_rng_state(z['rng'])
    write(p/'execution.json',execution)
    for epoch in range(start,epochs):
        ts=time.perf_counter();encoder.train(limited);head.train();order=train[torch.randperm(len(train),generator=g).numpy()];total=0
        for i in range(0,len(order),cfg['batch_size']):
            ix=order[i:i+cfg['batch_size']];x=encoder(e[ix],mk[ix]) if limited else emb[ix]
            pred=head((x-norm['xm'])/norm['xs']);loss=((pred-(target[ix]-norm['ym'])/norm['ys'])**2).mean()
            assert torch.isfinite(loss);o.zero_grad(set_to_none=True);loss.backward()
            variables=list(head.parameters())+[v for v in encoder.parameters() if v.requires_grad]
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in variables)
            nn.utils.clip_grad_norm_(variables,cfg['clip'],error_if_nonfinite=True);o.step();total+=float(loss.detach())*len(ix)
        log(p,{'epoch':epoch+1,'train_mse':total/len(train),'seconds':time.perf_counter()-ts})
        if epoch+1 in [epochs//2,epochs]:
            delta={k:v for k,v in encoder.state_dict().items() if not k.startswith('stem.')} if limited else {}
            torch.save({'head':head.state_dict(),'encoder_delta':delta,'norm':norm,'optimizer':o.state_dict(),'epoch':epoch+1,'generator':g.get_state(),'rng':torch.get_rng_state(),'execution':execution},p/f'epoch_{epoch+1:03d}.pt')
    head.eval();encoder.eval()
    with torch.inference_mode():
        out=torch.cat([head((encoder(e[ix],mk[ix])-norm['xm'])/norm['xs'])*norm['ys']+norm['ym'] for ix in [hold[i:i+128] for i in range(0,len(hold),128)]]).numpy().astype(np.float64)
    assert np.isfinite(out).all()
    np.savez_compressed(p/'oof.npz',image=meta['image'][hold],patient=meta['patient'][hold],site=meta['site'][hold],fold=meta['fold'][hold],row_index=hold,target=meta['target'][hold],prediction=out)
    delta={k:v for k,v in encoder.state_dict().items() if not k.startswith('stem.')} if limited else {}
    torch.save({'head':head.state_dict(),'encoder_delta':delta,'norm':norm,'execution':execution},p/'weights.pt')
    write(p/'verification.json',{'stem_unchanged':original_stem==state_sha(encoder.stem.state_dict()),'encoder_changed':original_encoder!=state_sha(encoder.state_dict()),'heldout_people_used_for_fit_or_normalization':False,'prediction_sha256':sha(p/'oof.npz'),'weights_sha256':sha(p/'weights.pt'),'heldout_images':len(hold),'heldout_people':4,'epochs':epochs})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','cache','pretrain','transfer']);p.add_argument('--config',required=True);p.add_argument('--freeze');p.add_argument('--initial');p.add_argument('--cache');p.add_argument('--source',choices=['stw','mskcc']);p.add_argument('--variant',choices=['B','C','D']);p.add_argument('--mode',choices=['frozen','limited']);p.add_argument('--fold',type=int);p.add_argument('--resume');p.add_argument('--out',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());torch.set_num_threads(c['threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    {'init':init,'cache':cache,'pretrain':pretrain,'transfer':transfer}[a.command](a,c)


if __name__=='__main__':main()
