"""Frozen UCI pixel model on a pre-hashed LaPa validation subset; CPU only."""
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import binary_mask, scores
from skin_lapa_prepare import file_sha
from threadpoolctl import threadpool_limits


def main():
    data = DATA_ROOT/'lapa'/'prepared_192'
    source = DATA_ROOT/'skin_pixel_v1'
    out = DATA_ROOT/'facial_skin_v1'
    if (out/'uci_validation_baseline.json').exists():
        raise RuntimeError('completed baseline exists; preserve it')
    model_result = json.loads((source/'results.json').read_text())
    raw = (source/'selected_local_model.pkl').read_bytes()
    if sha_bytes(raw) != model_result['model_sha256']:
        raise ValueError('own model checksum changed')
    model = pickle.loads(raw)
    threshold = json.loads((source/'selection.json').read_text())['winner']['threshold']
    names = json.loads((data/'val_order.json').read_text())
    ids = sorted(range(len(names)),key=lambda i:hashlib.sha256(('LumaSegVal128|'+names[i]).encode()).digest())[:128]
    protocol = dict(split='val',selection='smallest SHA256(LumaSegVal128|stem)',images=len(ids),
                     stems=[names[i] for i in ids],model_sha256=model_result['model_sha256'],threshold=threshold,
                     profile_sha256=file_sha(data/'profile.json'),code_sha256=sha_bytes(Path(__file__).read_bytes()))
    write_json(out/'uci_validation_protocol.json',protocol)
    images = np.load(data/'val_rgb.npy',mmap_mode='r')
    labels = np.load(data/'val_labels.npy',mmap_mode='r')
    confusion,rows = np.zeros(4,np.int64),[]
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for j,i in enumerate(ids):
            p = model.predict_proba(images[i][...,::-1].reshape(-1,3).astype(float)/255)[:,1].reshape(labels[i].shape) >= threshold
            truth = binary_mask(labels[i]).astype(bool)
            c = np.array([(p & truth).sum(),(p & ~truth).sum(),(~p & truth).sum(),(~p & ~truth).sum()])
            confusion += c
            rows.append(dict(stem=names[i],**scores(*c)))
            if (j+1) % 32 == 0:
                print('UCI LAPA VAL',j+1,'of',len(ids),flush=True)
    result = dict(**scores(*confusion),mean_image_iou=float(np.mean([r['iou'] for r in rows])),
                    images=len(rows),seconds=time.perf_counter()-started,rows=rows,
                    protocol_sha256=file_sha(out/'uci_validation_protocol.json'),
                    task='facial skin plus nose versus remaining LaPa classes; excludes neck/body skin',
                    measured_color_accuracy=None)
    write_json(out/'uci_validation_baseline.json',result)
    print('UCI LAPA VALIDATION RESULT',json.dumps({k:v for k,v in result.items() if k != 'rows'}),flush=True)


if __name__ == '__main__':
    main()
