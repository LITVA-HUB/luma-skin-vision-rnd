"""Prospective LaPa segmentation run with validation-only selection and live progress."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet, augment, scores, skin_loss
from skin_lapa_prepare import file_sha

ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_ROOT/'facial_skin_v1'
DATA = DATA_ROOT/'lapa'/'prepared_192'


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('RTX/CUDA required for this run')
    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    np.random.seed(20260914)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def batch(images,labels,ids):
    x = torch.from_numpy(np.array(images[ids],copy=True)).permute(0,3,1,2).to('cuda',dtype=torch.float32,memory_format=torch.channels_last)/255
    raw = np.array(labels[ids],copy=True)
    y = torch.from_numpy(((raw == 1)|(raw == 6)).astype(np.float32))[:,None].to('cuda')
    return x,y


@torch.inference_mode()
def evaluate(model,images,labels,batch_size=32):
    model.eval()
    confusion = torch.zeros(4,device='cuda',dtype=torch.int64)
    per_image = []
    for start in range(0,len(images),batch_size):
        x,y = batch(images,labels,slice(start,start+batch_size))
        with torch.autocast('cuda',dtype=torch.float16):
            pred = model(x) >= 0
        truth = y.bool()
        tp = (pred & truth).sum((1,2,3))
        fp = (pred & ~truth).sum((1,2,3))
        fn = (~pred & truth).sum((1,2,3))
        tn = (~pred & ~truth).sum((1,2,3))
        confusion += torch.stack([tp.sum(),fp.sum(),fn.sum(),tn.sum()])
        union = tp+fp+fn
        per_image.extend(torch.where(union>0,tp.double()/union,torch.ones_like(union,dtype=torch.double)).cpu().tolist())
    result = scores(*confusion.cpu().tolist())
    result.update(images=len(images),mean_image_iou=float(np.mean(per_image)),
                  median_image_iou=float(np.median(per_image)),
                  image_iou_p10=float(np.quantile(per_image,.1)))
    return result,np.asarray(per_image)


def require_prior_completion():
    seal = ROOT/'docs/benchmarks/chromaseed_architecture_scale_v1/verification.json'
    if not seal.exists():
        raise RuntimeError('AS audit/reconstruction/seal must finish before a new GPU job')
    subprocess.run([sys.executable,str(ROOT/'scripts/chromaseed_architecture_scale_report.py')],cwd=ROOT,check=True)


def preflight(batch_size):
    require_prior_completion()
    setup()
    model = SkinUNet().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    x = torch.rand(batch_size,3,192,192,device='cuda').to(memory_format=torch.channels_last)
    y = (torch.rand(batch_size,1,192,192,device='cuda') > .5).float()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss = skin_loss(model(x),y)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite preflight loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite preflight gradient')
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    receipt = dict(passed=True,batch_size=batch_size,steps=3,input_size=192,
                    parameters=sum(p.numel() for p in model.parameters()),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    seconds=time.perf_counter()-started,device=torch.cuda.get_device_name(0),
                    torch_version=str(torch.__version__),
                    sources={p.name:sha_bytes(p.read_bytes()) for p in [Path(__file__),ROOT/'scripts/skin_face_segment.py']})
    write_json(OUT/'preflight.json',receipt)
    print('SEG PREFLIGHT',json.dumps(receipt),flush=True)


def train():
    require_prior_completion()
    pre = json.loads((OUT/'preflight.json').read_text())
    for name,digest in pre['sources'].items():
        if file_sha(ROOT/'scripts'/name) != digest:
            raise ValueError('source changed after preflight')
    if (OUT/'protocol.json').exists():
        raise RuntimeError('training protocol already exists; preserve this run')
    profile = json.loads((DATA/'profile.json').read_text())
    for split,record in profile['splits'].items():
        for kind in ['rgb','labels']:
            if file_sha(DATA/f'{split}_{kind}.npy') != record[kind+'_sha256']:
                raise ValueError('prepared dataset checksum changed')
    setup()
    arrays = {s:{k:np.load(DATA/f'{s}_{k}.npy',mmap_mode='r',allow_pickle=False) for k in ['rgb','labels']} for s in ['train','val']}
    epochs,batch_size = 12,pre['batch_size']
    sources = [Path(__file__),ROOT/'scripts/skin_face_segment.py',ROOT/'scripts/skin_lapa_prepare.py',
               ROOT/'scripts/skin_face_evaluate.py',ROOT/'scripts/skin_face_baseline.py',
               ROOT/'tests/test_skin_face_segment.py',ROOT/'docs/superpowers/plans/2026-09-14-facial-skin.md']
    protocol = dict(name='Luma ChromaSeed-Seg1',seed=20260914,width=24,size=192,epochs=epochs,
                     batch_size=batch_size,optimizer='AdamW',learning_rate=.0003,weight_decay=.0001,
                     loss='BCEWithLogits + per-image soft Dice',positive_labels=[1,6],logit_threshold=0,
                     augmentation='train only: 50% horizontal flip, channel gain .9..1.1, exposure/gamma .85..1.15',
                     schedule='cosine per optimizer step; multiplier 1.0 to 0.1',
                     selection='max validation mean per-image IoU; earliest epoch wins ties',
                     test_accessed=False,test_comparison_subset='128 official test image names with smallest SHA256(LumaSegTest128|stem), after exclusions; selected without decoding',
                     no_measured_color_claim=True,profile_sha256=file_sha(DATA/'profile.json'),
                     sources={str(p.relative_to(ROOT)):file_sha(p) for p in sources},
                     preflight_sha256=file_sha(OUT/'preflight.json'),
                     reproducibility='fixed random seeds; cuDNN autotuning/AMP enabled, no bitwise cross-device training claim',
                     started_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'job.json',dict(pid=os.getpid(),stage='training',started_utc=protocol['started_utc']))
    model = SkinUNet().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    rng = np.random.default_rng(20260914)
    steps_per_epoch = math.ceil(len(arrays['train']['rgb'])/batch_size)
    total_steps,global_step = steps_per_epoch*epochs,0
    best,history = -1.,[]
    start_time,last_update = time.perf_counter(),time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        order = rng.permutation(len(arrays['train']['rgb']))
        total_loss,seen = 0.,0
        epoch_start = time.perf_counter()
        for offset in range(0,len(order),batch_size):
            factor = .1+.9*.5*(1+math.cos(math.pi*global_step/max(1,total_steps-1)))
            for group in optimizer.param_groups:
                group['lr'] = .0003*factor
            ids = order[offset:offset+batch_size]
            x,y = batch(arrays['train']['rgb'],arrays['train']['labels'],ids)
            x,y = augment(x,y)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss = skin_loss(model(x),y)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())*len(ids)
            seen += len(ids)
            global_step += 1
            elapsed = time.perf_counter()-start_time
            if time.perf_counter()-last_update > 10 or offset+batch_size >= len(order):
                progress = dict(status='training',epoch=epoch,epochs=epochs,step=global_step,total_steps=total_steps,
                                images_seen=(epoch-1)*len(order)+seen,elapsed_seconds=elapsed,
                                images_per_second=seen/(time.perf_counter()-epoch_start),
                                estimated_remaining_seconds=elapsed/max(1,global_step)*(total_steps-global_step),
                                training_loss=total_loss/seen,best_validation_mean_iou=best if best>=0 else None,
                                updated_utc=datetime.now(timezone.utc).isoformat())
                write_json(OUT/'progress.json',progress)
                print('SEG PROGRESS',json.dumps(progress),flush=True)
                last_update = time.perf_counter()
        validation,per_image = evaluate(model,arrays['val']['rgb'],arrays['val']['labels'],batch_size)
        row = dict(epoch=epoch,training_loss=total_loss/seen,validation=validation,seconds=time.perf_counter()-epoch_start)
        history.append(row)
        np.save(OUT/f'validation_epoch_{epoch:02d}_iou.npy',per_image,allow_pickle=False)
        if validation['mean_image_iou'] > best:
            best = validation['mean_image_iou']
            state = {key:value.detach().cpu() for key,value in model.state_dict().items()}
            torch.save(dict(state_dict=state,width=24,epoch=epoch),OUT/'best.pt')
            write_json(OUT/'best_validation.json',row)
        write_json(OUT/'history.json',history)
        print('SEG EPOCH',json.dumps(row),flush=True)
    elapsed = time.perf_counter()-start_time
    selected = json.loads((OUT/'best_validation.json').read_text())
    selection = dict(selected_epoch=selected['epoch'],validation=selected['validation'],
                      model_sha256=file_sha(OUT/'best.pt'),model_bytes=(OUT/'best.pt').stat().st_size,
                      protocol_sha256=file_sha(OUT/'protocol.json'),history_sha256=file_sha(OUT/'history.json'),
                      workflow_seconds=elapsed,parameters=pre['parameters'],test_accessed=False)
    write_json(OUT/'selection.json',selection)
    write_json(OUT/'progress.json',dict(status='training_complete_pending_test',epoch=epochs,epochs=epochs,
               step=global_step,total_steps=total_steps,elapsed_seconds=elapsed,estimated_remaining_seconds=0,
               best_validation_mean_iou=best,updated_utc=datetime.now(timezone.utc).isoformat()))
    print('SEG TRAINING COMPLETE',json.dumps(selection),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--batch',type=int,default=32)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.batch)
    if args.train:
        train()


if __name__ == '__main__':
    main()
