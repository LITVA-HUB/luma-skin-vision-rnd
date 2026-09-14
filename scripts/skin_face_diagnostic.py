"""CPU-only qualitative overlays for supplied photos and the public DAST example."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet
from skin_lapa_prepare import file_sha


def main():
    run = DATA_ROOT/'facial_skin_v1'
    result = json.loads((run/'test_results.json').read_text())
    if file_sha(run/'best.pt') != result['model_sha256']:
        raise ValueError('model checksum changed')
    torch.set_num_threads(2)
    checkpoint = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model = SkinUNet(width=checkpoint['width']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    private = DATA_ROOT/'private_user_faces'
    photos = json.loads((private/'manifest.json').read_text(encoding='utf-8'))['records']
    dast_root = DATA_ROOT/'dast_public_example'
    dast = json.loads((dast_root/'profile.json').read_text())['images']
    rows = []
    contact = Image.new('RGB',(1200,680),'#f1f6f3')
    draw = ImageDraw.Draw(contact)
    draw.text((12,8),'Supplied photos / ChromaSeed-Seg1 facial skin mask (CPU FP32)',fill='#22372c')
    with torch.inference_mode():
        for i,item in enumerate(photos+dast):
            path = Path(item['path'])
            if file_sha(path) != item['sha256']:
                raise ValueError('diagnostic input checksum changed')
            with Image.open(path) as source:
                source.load()
                im = source.convert('RGB')
            rgb = np.asarray(im.resize((192,192),Image.Resampling.BILINEAR)).copy()
            x = torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
            probability = torch.sigmoid(model(x))[0,0].numpy()
            mask = probability >= .5
            is_private = i < len(photos)
            destination = private/'seg1_masks' if is_private else dast_root/'seg1_masks'
            destination.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask.astype(np.uint8)*255).save(destination/(item['sha256']+'.png'))
            rows.append(dict(source='user_supplied' if is_private else 'DAST_public_example',
                             image_sha256=item['sha256'],subject_id=item.get('subject_id'),
                             mask_fraction_192=float(mask.mean()),ground_truth_lab=None,
                             semantic_accuracy=None,note='predicted mask only; no manually annotated full reference mask'))
            if is_private:
                thumb = im.resize((240,320),Image.Resampling.BILINEAR)
                selected = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((240,320),Image.Resampling.NEAREST)).astype(bool)
                view = np.asarray(thumb).astype(float)
                view[selected] = view[selected]*.45+np.array([25,235,174])*.55
                contact.paste(thumb,(i*240,30))
                contact.paste(Image.fromarray(view.astype(np.uint8)),(i*240,355))
    contact.save(private/'seg1_diagnostic.png')
    write_json(run/'diagnostic.json',dict(rows=rows,model_sha256=result['model_sha256'],
                 code_sha256=sha_bytes(Path(__file__).read_bytes()),precision='CPU float32',
                 photo_accuracy=None,lab_accuracy=None,
                 limitation='Qualitative deployment examples only. No training or threshold selection on these photos.'))
    print('SEG DIAGNOSTICS',len(photos),'user photos,',len(dast),'DAST portraits; no color accuracy inferred',flush=True)


if __name__ == '__main__':
    main()
