"""First held-image evaluation only after the segmentation checkpoint is frozen."""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet, binary_mask, scores
from skin_face_train import DATA, OUT, ROOT, evaluate, setup
from skin_lapa_prepare import file_sha
from threadpoolctl import threadpool_limits


def main():
    if (OUT/'test_results.json').exists():
        raise RuntimeError('held evaluation exists; preserve it')
    selection = json.loads((OUT/'selection.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    if file_sha(OUT/'best.pt') != selection['model_sha256'] or file_sha(OUT/'protocol.json') != selection['protocol_sha256']:
        raise ValueError('selection/model changed before held evaluation')
    for path,digest in protocol['sources'].items():
        if file_sha(ROOT/path) != digest:
            raise ValueError('bound experiment source changed')
    original_root = DATA_ROOT/'lapa'
    all_rows = json.loads((original_root/'usable_triplets.json').read_text())
    rows = [r for r in all_rows if r['split'] == 'test']
    ids = sorted(range(len(rows)),key=lambda i:hashlib.sha256(('LumaSegTest128|'+rows[i]['stem']).encode()).digest())[:128]
    test_protocol = dict(selection_sha256=file_sha(OUT/'selection.json'),
                         triplets_sha256=file_sha(original_root/'usable_triplets.json'),
                         test_images=len(rows),baseline_subset=[rows[i]['stem'] for i in ids],
                         code_sha256=sha_bytes(Path(__file__).read_bytes()),
                         test_decoded_before_this_protocol=False)
    write_json(OUT/'test_protocol.json',test_protocol)
    write_json(OUT/'job.json',dict(pid=os.getpid(),stage='evaluation'))
    write_json(OUT/'progress.json',dict(status='evaluating',estimated_remaining_seconds=None))
    images = np.empty((len(rows),192,192,3),np.uint8)
    labels = np.empty((len(rows),192,192),np.uint8)
    started = time.perf_counter()
    for i,row in enumerate(rows):
        c = row['components']
        for kind in ['images','labels']:
            if file_sha(c[kind]['path']) != c[kind]['sha256']:
                raise ValueError('test original checksum changed')
        with Image.open(c['images']['path']) as im,Image.open(c['labels']['path']) as lab:
            im.load()
            lab.load()
            raw = np.asarray(lab)
            if im.size != lab.size or raw.ndim != 2 or not np.isin(raw,np.arange(11)).all():
                raise ValueError('invalid held image/label pair')
            images[i] = np.asarray(im.convert('RGB').resize((192,192),Image.Resampling.BILINEAR))
            labels[i] = np.asarray(lab.resize((192,192),Image.Resampling.NEAREST))
    # Validate the held decoder against the already frozen validation preprocessing.
    val_rows = [r for r in all_rows if r['split'] == 'val']
    val_rgb = np.load(DATA/'val_rgb.npy',mmap_mode='r')
    val_labels = np.load(DATA/'val_labels.npy',mmap_mode='r')
    for i in [0, len(val_rows)//2, len(val_rows)-1]:
        with Image.open(val_rows[i]['components']['images']['path']) as im,Image.open(val_rows[i]['components']['labels']['path']) as lab:
            np.testing.assert_array_equal(np.asarray(im.convert('RGB').resize((192,192),Image.Resampling.BILINEAR)),val_rgb[i])
            np.testing.assert_array_equal(np.asarray(lab.resize((192,192),Image.Resampling.NEAREST)),val_labels[i])
    setup()
    checkpoint = torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint['epoch'] != selection['selected_epoch'] or checkpoint['width'] != protocol['width']:
        raise ValueError('checkpoint metadata disagrees with selection')
    if not all(torch.isfinite(p).all() for p in checkpoint['state_dict'].values()):
        raise ValueError('nonfinite saved parameters')
    model = SkinUNet(width=protocol['width']).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    held,per_image = evaluate(model,images,labels,protocol['batch_size'])
    subset,subset_iou = evaluate(model,images[ids],labels[ids],protocol['batch_size'])
    np.savez_compressed(OUT/'test_image_metrics.npz',stems=np.asarray([r['stem'] for r in rows]),iou=per_image,
                        comparison_indices=np.asarray(ids),comparison_iou=subset_iou)
    print('SEG HELD CNN',json.dumps(held),flush=True)
    baseline_root = DATA_ROOT/'skin_pixel_v1'
    baseline_result = json.loads((baseline_root/'results.json').read_text())
    raw = (baseline_root/'selected_local_model.pkl').read_bytes()
    if sha_bytes(raw) != baseline_result['model_sha256']:
        raise ValueError('own frozen UCI model checksum changed')
    baseline = pickle.loads(raw)
    threshold = json.loads((baseline_root/'selection.json').read_text())['winner']['threshold']
    counts,baseline_rows = np.zeros(4,np.int64),[]
    with threadpool_limits(limits=1):
        for j,i in enumerate(ids):
            pred = baseline.predict_proba(images[i][...,::-1].reshape(-1,3).astype(float)/255)[:,1].reshape(192,192) >= threshold
            truth = binary_mask(labels[i]).astype(bool)
            c = np.array([(pred&truth).sum(),(pred&~truth).sum(),(~pred&truth).sum(),(~pred&~truth).sum()])
            counts += c
            baseline_rows.append(dict(stem=rows[i]['stem'],**scores(*c)))
            if (j+1) % 32 == 0:
                print('SEG HELD UCI',j+1,'of',len(ids),flush=True)
    base_scores = dict(**scores(*counts),mean_image_iou=float(np.mean([r['iou'] for r in baseline_rows])),images=len(ids))
    write_json(OUT/'test_baseline_per_image.json',baseline_rows)
    # Fixed observed-brightness strata are diagnostics, never model/threshold selectors.
    # Encoded sRGB brightness combines lighting and appearance; it is not intrinsic skin tone.
    brightness = []
    for rgb,lab in zip(images,labels,strict=True):
        skin = (lab == 1)|(lab == 6)
        value = float(np.mean((rgb[skin].astype(float)/255) @ np.array([.2126,.7152,.0722]))) if skin.any() else np.nan
        brightness.append(value)
    brightness = np.asarray(brightness)
    brightness_groups = []
    for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.000001)]:
        included = (brightness >= lo)&(brightness < hi)
        brightness_groups.append(dict(minimum=lo,maximum=min(hi,1.),images=int(included.sum()),
                                      mean_image_iou=float(per_image[included].mean()) if included.any() else None))
    # Independent CPU deserialization of our own saved model must preserve predictions exactly.
    a,b = SkinUNet(width=24).eval(),SkinUNet(width=24).eval()
    a.load_state_dict(checkpoint['state_dict'])
    b.load_state_dict(torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    x = torch.from_numpy(images[:2].copy()).permute(0,3,1,2).float()/255
    with torch.inference_mode():
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    result = dict(model='Luma ChromaSeed-Seg1',parameters=selection['parameters'],model_bytes=selection['model_bytes'],
                    selected_epoch=selection['selected_epoch'],test=held,paired_128=dict(cnn=subset,uci_color_only=base_scores),
                    protocol_sha256=file_sha(OUT/'test_protocol.json'),selection_sha256=file_sha(OUT/'selection.json'),
                    model_sha256=selection['model_sha256'],seconds=time.perf_counter()-started,
                    cpu_reload_exact=True,test_decoded=True,ground_truth_color_accuracy=None,
                    observed_srgb_brightness_groups=brightness_groups,
                    empty_ground_truth_images=int(np.isnan(brightness).sum()),
                    limitations=['semantic mask quality only; no measured color target',
                                 'official held images with source-prefix/byte overlap exclusions; person independence unverified',
                                 '192x192 evaluation; source-resolution boundaries not evaluated'])
    write_json(OUT/'test_results.json',result)
    write_json(OUT/'progress.json',dict(status='complete',epoch=12,epochs=12,estimated_remaining_seconds=0,
                  test_mean_image_iou=held['mean_image_iou'],best_validation_mean_iou=selection['validation']['mean_image_iou']))
    print('SEG TEST COMPLETE',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
