"""Seg2 mask and apparent-image color evaluation, after all validation choices are frozen."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from chromaseed_head_range_verification import process_alive
from skin_face_appearance import appearance, color_error
from skin_face_transfer_data import SOURCES, checked, digest, load_split, read
from skin_face_transfer_run import (
    load_checkpoint,
    precision,
    require_predecessors,
    save_npz,
    setup,
    tensor_batch,
)
from skin_face_transfer_study import (
    BATCH,
    COUNTS,
    INITIAL,
    INITIAL_SHA,
    RUN,
    check_bindings,
    freeze_choices,
    recipe,
    utc,
    verify_registration,
    write_once,
)


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.ones_like(numerator, dtype=np.float64), where=denominator>0)


def mask_summary(confusion):
    a = np.asarray(confusion)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.issubdtype(a.dtype,np.integer) or (a<0).any():
        raise ValueError('Nonempty nonnegative integer image confusion counts required')
    tp,fp,fn,tn = a.T
    iou = _ratio(tp,tp+fp+fn)
    dice = _ratio(2*tp,2*tp+fp+fn)
    precision = _ratio(tp,tp+fp)
    recall = _ratio(tp,tp+fn)
    totals = a.sum(axis=0)
    t,p,n,_ = totals
    return dict(images=len(a), confusion=totals.tolist(), global_iou=float(_ratio(t,t+p+n)),
                global_dice=float(_ratio(2*t,2*t+p+n)), global_precision=float(_ratio(t,t+p)),
                global_recall=float(_ratio(t,t+n)), mean_image_iou=float(iou.mean()),
                median_image_iou=float(np.median(iou)), image_iou_p10=float(np.quantile(iou,.1)),
                mean_image_dice=float(dice.mean()), mean_image_precision=float(precision.mean()),
                mean_image_recall=float(recall.mean()), fraction_image_iou_below_0_5=float((iou<.5).mean()))


def color_arrays(rgb, prediction, truth):
    n = len(rgb)
    result = {k:np.full((n,2,3), np.nan) for k in ('reference_lab','prediction_lab')}
    result['color_error'] = np.full((n,2),np.nan)
    result['reference_pixels'] = truth.sum(axis=(1,2),dtype=np.int64)
    result['prediction_pixels'] = prediction.sum(axis=(1,2),dtype=np.int64)
    for index,(image,pred,actual) in enumerate(zip(rgb,prediction,truth,strict=True)):
        reference, predicted = appearance(image,actual), appearance(image,pred)
        for key,value in (('reference_lab',reference),('prediction_lab',predicted)):
            if value is not None:
                result[key][index] = [value['median_lab'],value['mean_lab']]
        error = color_error(reference,predicted)
        if error is not None:
            result['color_error'][index] = [error['median_delta_e00'],error['mean_delta_e00']]
    return result


def color_summary(arrays):
    errors = arrays['color_error']
    valid = np.isfinite(errors).all(axis=1)
    return dict(total_images=len(errors), valid_references=int((arrays['reference_pixels']>=16).sum()),
                valid_predictions=int((arrays['prediction_pixels']>=16).sum()), available_pairs=int(valid.sum()),
                median_delta_e00_mean=float(errors[valid,0].mean()) if valid.any() else None,
                mean_delta_e00_mean=float(errors[valid,1].mean()) if valid.any() else None,
                physical_skin_truth=False)


def summary(arrays):
    result = mask_summary(arrays['confusion'])
    if 'color_error' in arrays:
        result['color'] = color_summary(arrays)
    return result


def confusion_from_masks(prediction, truth):
    if (prediction.dtype != bool or truth.dtype != bool or prediction.ndim != 3
            or prediction.shape != truth.shape):
        raise ValueError('Matching batch of boolean masks required')
    return np.stack([(prediction&truth).sum((1,2)), (prediction&~truth).sum((1,2)),
                     (~prediction&truth).sum((1,2)), (~prediction&~truth).sum((1,2))],axis=1).astype(np.int64)


def unpack_masks(arrays):
    shape = arrays['mask_shape']
    packed = arrays['packed_masks']
    if (shape.shape != (2,) or not np.issubdtype(shape.dtype,np.integer) or (shape<1).any()
            or packed.dtype != np.uint8 or packed.ndim != 2
            or packed.shape != (len(arrays['global_ids']), (int(np.prod(shape))+7)//8)):
        raise ValueError('Invalid packed mask dimensions or type')
    return np.unpackbits(packed,axis=1,count=int(np.prod(shape)),bitorder='little').reshape(-1,*shape).astype(bool)


def verify_mask_arrays(arrays, truth):
    prediction = unpack_masks(arrays)
    actual = confusion_from_masks(prediction,truth)
    if not np.array_equal(actual,arrays['confusion']):
        raise ValueError('Saved confusion differs from actual mask pixels')
    return prediction


@torch.inference_mode()
def evaluate_split(model, split, batch_size, device, include_color=False):
    if split.role not in ('validation','test'):
        raise ValueError('Evaluation requires a validation or test view')
    if not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('Positive batch size required')
    model.eval()
    chunks, spatial = {}, None
    for start in range(0,len(split.indices),batch_size):
        ids = split.indices[start:start+batch_size]
        rgb,labels = split.get(ids)
        x,y = tensor_batch(rgb,labels,device)
        with precision(device):
            logits = model(x)
        if logits.shape != y.shape or not torch.isfinite(logits).all():
            raise ValueError('Invalid mask model output')
        predicted = (logits[:,0]>=0).cpu().numpy()
        truth = labels.astype(bool)
        spatial = np.asarray(truth.shape[1:],dtype=np.int64)
        current = dict(global_ids=ids, confusion=confusion_from_masks(predicted,truth),
                       packed_masks=np.packbits(predicted.reshape(len(ids),-1),axis=1,bitorder='little'))
        if include_color:
            current.update(color_arrays(rgb,predicted,truth))
        for name,value in current.items():
            chunks.setdefault(name,[]).append(value)
    arrays = {k:np.concatenate(v) for k,v in chunks.items()}
    arrays['mask_shape'] = spatial
    return summary(arrays), arrays


def validate_selection(selected, histories):
    expected = freeze_choices(histories)
    if any(selected.get(key) != value for key,value in expected.items()) or 'test' in selected:
        raise ValueError('Frozen selection differs from complete validation-only recomputation')


def completed_selection():
    verify_registration()
    if not (RUN/'completion.json').exists() or not (RUN/'selection.json').exists():
        raise RuntimeError('All six trajectories and frozen choices must complete first')
    completed, selected = read(RUN/'completion.json'), read(RUN/'selection.json')
    if completed.get('status') != 'complete' or process_alive(completed.get('pid')) is not False:
        raise RuntimeError('Seg2 training worker must be complete and confirmed dead')
    registration_sha = digest(RUN/'registration.json')
    if completed['registration_sha256'] != registration_sha or selected['registration_sha256'] != registration_sha:
        raise ValueError('Training and selection refer to a different registration')
    checked(RUN/'selection.json',completed['selection_sha256'])
    check_bindings(selected['histories'])
    spec = recipe()
    histories = {}
    if [r['id'] for r in completed['trajectories']] != [r['id'] for r in spec['trajectories']]:
        raise ValueError('Missing or reordered completed trajectories')
    for record, wanted in zip(completed['trajectories'],spec['trajectories'],strict=True):
        if any(record.get(k) != v for k,v in wanted.items()):
            raise ValueError('Completed trajectory no longer matches the recipe')
        path = RUN/'trajectories'/record['id']/'history.json'
        if record['history_path'] != str(path):
            raise ValueError('History path differs from its registered trajectory')
        checked(path,record['history_sha256'])
        histories[record['id']] = read(path)
    if selected['histories'] != {r['history_path']:r['history_sha256'] for r in completed['trajectories']}:
        raise ValueError('Selection history binding mismatch')
    validate_selection(selected,histories)
    for history in histories.values():
        for row in history:
            checked(row['checkpoint']['path'],row['checkpoint']['sha256'])
    return selected, completed, histories


def selected_models(selection):
    models = {INITIAL_SHA:dict(path=str(INITIAL),sha256=INITIAL_SHA)}
    for record in selection['choices']:
        descriptor = record['checkpoint']
        sha = descriptor['sha256']
        if sha in models and models[sha]['path'] != descriptor['path']:
            raise ValueError('A selected checkpoint hash has conflicting paths')
        models[sha] = descriptor
    return models


def run_held():
    require_predecessors()
    selection, completed, histories = completed_selection()
    if (RUN/'test_protocol.json').exists() or (RUN/'test_results.json').exists():
        raise FileExistsError('Preserve existing held evaluation; no automatic repeat or reselection')
    from skin_face_transfer_report import verify_training_artifacts
    training_audit = verify_training_artifacts(completed,histories)
    write_once(RUN/'training_audit.json',training_audit)
    models = selected_models(selection)
    protocol = dict(created_utc=utc(), pid=os.getpid(), selection_sha256=digest(RUN/'selection.json'),
                    completion_sha256=digest(RUN/'completion.json'), training_audit_sha256=digest(RUN/'training_audit.json'),
                    registration_sha256=digest(RUN/'registration.json'), models=models,
                    source_counts={s:COUNTS[s]['test'] for s in SOURCES},
                    lapa_test_exposed_before_study=True, celeba_test_neural_inference_before_study=False,
                    precision='CUDA FP16 autocast; batch32; logit>=0', apparent_color_only=True,
                    measured_color_targets=False, minimum_color_pixels=16,
                    common_color_coverage='pair-specific intersections; also show marginal missingness')
    write_once(RUN/'test_protocol.json',protocol)
    started = time.perf_counter()
    setup('cuda')
    splits = {s:load_split(s,'test',allow_test=True) for s in SOURCES}
    for source,split in splits.items():
        if len(split.indices) != COUNTS[source]['test']:
            raise ValueError('Held source count mismatch')
        write_once(RUN/f'test_{source}_metadata.json', [dict(global_id=int(i),**split.metadata[i]) for i in split.indices])
    rows = []
    for sha,descriptor in models.items():
        model = load_checkpoint(descriptor,'cuda').to(memory_format=torch.channels_last)
        for source,split in splits.items():
            metric,arrays = evaluate_split(model,split,BATCH,'cuda',include_color=True)
            array_ref = save_npz(RUN/'test'/f'{sha}_{source}.npz',**arrays)
            record = dict(checkpoint_sha256=sha,source=source,metrics=metric,arrays=array_ref)
            write_once(RUN/'test'/f'{sha}_{source}.json',record)
            rows.append(record)
            print('SEG2 TEST',sha,source,metric['mean_image_iou'],flush=True)
        del model
    checked(RUN/'selection.json',protocol['selection_sha256'])
    result = dict(status='complete', pid=os.getpid(), records=rows, test_protocol_sha256=digest(RUN/'test_protocol.json'),
                  selection_sha256=protocol['selection_sha256'], elapsed_seconds=time.perf_counter()-started,
                  finished_utc=utc(), measured_color_accuracy=None)
    write_once(RUN/'test_results.json',result)
    print('SEG2 HELD EVALUATION COMPLETE',len(rows),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['test'])
    parser.parse_args()
    run_held()


if __name__ == '__main__':
    main()
