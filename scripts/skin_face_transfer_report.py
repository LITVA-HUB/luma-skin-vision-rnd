"""Complete Seg2 artifact checks, paired comparisons and quiet-host CPU response timings."""
from __future__ import annotations

import argparse
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from skin_face_transfer_data import SOURCES, PairedSampler, checked, digest, load_split, read
from skin_face_transfer_evaluate import (
    color_arrays,
    completed_selection,
    selected_models,
    summary,
    unpack_masks,
    verify_mask_arrays,
)
from skin_face_transfer_run import (
    load_checkpoint,
    load_training_views,
    require_finished,
    require_predecessors,
    tensor_batch,
)
from skin_face_transfer_study import (
    ARMS,
    BATCH,
    BUDGETS,
    INITIAL_SHA,
    PARAMETERS,
    RUN,
    SEEDS,
    check_bindings,
    recipe,
    utc,
    write_once,
)


def arrays_from(reference):
    checked(reference['path'],reference['sha256'])
    with np.load(reference['path'],allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def paired_color(left,right):
    if not np.array_equal(left['global_ids'],right['global_ids']):
        raise ValueError('Pairwise color requires identical source row order')
    a,b = left['color_error'],right['color_error']
    av,bv = np.isfinite(a).all(axis=1),np.isfinite(b).all(axis=1)
    common = av&bv
    return dict(common_images=int(common.sum()), left_available=int(av.sum()),right_available=int(bv.sum()),
                right_minus_left_median_delta_e00=float((b[common,0]-a[common,0]).mean()) if common.any() else None,
                right_minus_left_mean_delta_e00=float((b[common,1]-a[common,1]).mean()) if common.any() else None)


def validate_trace(trace,arm,seed,lapa_ids,celeba_ids,steps=BUDGETS[-1],batch_size=BATCH):
    ids,sources,values = trace['global_ids'],trace['source_ids'],trace['loss_gradient_lr']
    if (ids.shape != (steps,batch_size) or sources.shape != ids.shape or values.shape != (steps,3)
            or ids.dtype != np.int64 or sources.dtype != np.uint8 or values.dtype != np.float64
            or not np.isfinite(values).all() or (values<0).any()):
        raise ValueError('Invalid or incomplete training trace')
    sampler = PairedSampler(lapa_ids,celeba_ids,seed,batch_size)
    for step in range(steps):
        expected_sources,expected_ids = sampler.next(arm)
        rate = .0001*(.1+.9*.5*(1+math.cos(math.pi*step/(BUDGETS[-1]-1))))
        if (not np.array_equal(sources[step],expected_sources) or not np.array_equal(ids[step],expected_ids)
                or abs(values[step,2]-rate)>1e-18):
            raise ValueError('Training trace differs from the paired TRAIN-only stream or schedule')
    return sampler.cross_stream_duplicate_presentations


def verify_array_record(metric,split,include_color):
    arrays = arrays_from(metric['arrays'])
    if not np.array_equal(arrays['global_ids'],split.indices):
        raise ValueError('Evaluation row order differs from the source split')
    if summary(arrays) != {k:v for k,v in metric.items() if k != 'arrays'}:
        raise ValueError('Stored evaluation summary differs from per-image arrays')
    prediction = unpack_masks(arrays)
    for start in range(0,len(split.indices),128):
        end = min(len(split.indices),start+128)
        rgb,truth = split.get(split.indices[start:end])
        part = {k:(v if k == 'mask_shape' else v[start:end]) for k,v in arrays.items()}
        verify_mask_arrays(part,truth.astype(bool))
        if include_color:
            expected = color_arrays(rgb,prediction[start:end],truth.astype(bool))
            for name,value in expected.items():
                if not np.allclose(value,part[name],rtol=0,atol=1e-12,equal_nan=True):
                    raise ValueError(f'Apparent color differs from saved mask/source image: {name}')
    return arrays


def verify_training_artifacts(completion,histories):
    if completion['successful_updates'] != recipe()['new_updates']:
        raise ValueError('Incomplete total training budget')
    training = load_training_views()
    validation = {s:load_split(s,'validation') for s in SOURCES}
    baseline = read(RUN/'baseline/validation.json')
    checked(baseline['checkpoint']['path'],INITIAL_SHA)
    artifacts, validated = {}, set()
    def remember(path):
        artifacts[str(path)] = digest(path)
    remember(RUN/'baseline/validation.json')
    for receipt in completion['trajectories']:
        directory = RUN/'trajectories'/receipt['id']
        if read(directory/'receipt.json') != receipt:
            raise ValueError('Completion record differs from trajectory receipt')
        if receipt['attempted_batches'] != BUDGETS[-1] or receipt['successful_updates'] != BUDGETS[-1]:
            raise ValueError('Attempted and successful update counts differ from fixed budget')
        trace = arrays_from(receipt['sample_trace'])
        duplicates = validate_trace(trace,receipt['arm'],receipt['seed'],training[0].indices,training[1].indices)
        if duplicates != receipt['cross_stream_duplicate_presentations']:
            raise ValueError('Cross-stream duplicate count differs from actual samples')
        for path in (directory/'receipt.json',directory/'history.json',receipt['sample_trace']['path']):
            remember(path)
        if histories[receipt['id']][0] != baseline:
            raise ValueError('A trajectory has a different unchanged baseline')
        for row in histories[receipt['id']]:
            if row['step']:
                path = directory/f"step_{row['step']:04d}.json"
                if read(path) != row:
                    raise ValueError('Checkpoint receipt differs from final history')
                remember(path)
            checked(row['checkpoint']['path'],row['checkpoint']['sha256'])
            remember(row['checkpoint']['path'])
            for source,metric in row['validation'].items():
                key = metric['arrays']['path']
                if key not in validated:
                    verify_array_record(metric,validation[source],False)
                    validated.add(key)
                    remember(key)
    return dict(passed=True, method='complete sample-stream replay and saved-mask/source-pixel arithmetic; no training or neural inference replay',
                trajectories=6, updates=recipe()['new_updates'], validation_array_files=len(validated),
                artifacts=artifacts, test_accessed=False, created_utc=utc())


def audit():
    require_predecessors()
    selection,_,_ = completed_selection()
    if (RUN/'test_audit.json').exists():
        raise FileExistsError('Preserve existing held audit')
    protocol,results = read(RUN/'test_protocol.json'),require_finished(RUN/'test_results.json')
    checked(RUN/'test_protocol.json',results['test_protocol_sha256'])
    checked(RUN/'selection.json',protocol['selection_sha256'])
    checked(RUN/'training_audit.json',protocol['training_audit_sha256'])
    check_bindings(read(RUN/'training_audit.json')['artifacts'])
    models = selected_models(selection)
    if models != protocol['models'] or results['selection_sha256'] != protocol['selection_sha256']:
        raise ValueError('Held evaluation model set differs from validation selection')
    expected = {(sha,source) for sha in models for source in SOURCES}
    actual = [(r['checkpoint_sha256'],r['source']) for r in results['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing, extra or duplicated held model/source outcomes')
    splits = {s:load_split(s,'test',allow_test=True) for s in SOURCES}
    artifacts = {str(RUN/name):digest(RUN/name) for name in (
        'registration.json','completion.json','selection.json','test_protocol.json','test_results.json','training_audit.json')}
    count = 0
    for record in results['records']:
        arrays = verify_array_record({**record['metrics'],'arrays':record['arrays']},splits[record['source']],True)
        count += len(arrays['global_ids'])
        artifacts[record['arrays']['path']] = record['arrays']['sha256']
    value = dict(status='complete', pid=os.getpid(), passed=True, records=len(actual), image_predictions=count, artifacts=artifacts, created_utc=utc(),
                 method='source/checkpoint hashes; complete saved-mask confusion and apparent-color reconstruction; no independent network inference replay',
                 ordinary_phone_instrument_accuracy=None)
    write_once(RUN/'test_audit.json',value)
    print('SEG2 TEST AUDIT PASSED',len(actual),count,flush=True)


def runtime():
    require_predecessors()
    selection,_,_ = completed_selection()
    audit_value = require_finished(RUN/'test_audit.json')
    if audit_value.get('passed') is not True:
        raise ValueError('Held audit must pass before runtime measurements')
    check_bindings(audit_value['artifacts'])
    if (RUN/'runtime_protocol.json').exists():
        raise FileExistsError('Preserve existing runtime attempt')
    torch.set_num_threads(2)
    split = load_split('lapa','validation')
    rgb,labels = split.get(split.indices[:1])
    x,_ = tensor_batch(rgb,labels,'cpu')
    protocol = dict(created_utc=utc(), pid=os.getpid(), threads=2, batch_size=1, input_size=192,
                    device='CPU', precision='FP32', memory_format='channels_last',
                    warmup_calls=10, measured_calls=60, source='first registered LaPa validation row',
                    timing_scope='model forward only; excludes image decoding, resizing and color extraction',
                    hardware=platform.processor(), system=platform.platform(), torch_version=str(torch.__version__),
                    selection_sha256=digest(RUN/'selection.json'),test_audit_sha256=digest(RUN/'test_audit.json'))
    write_once(RUN/'runtime_protocol.json',protocol)
    rows = []
    with torch.inference_mode():
        for sha,descriptor in selected_models(selection).items():
            model = load_checkpoint(descriptor).to(memory_format=torch.channels_last)
            for _ in range(protocol['warmup_calls']):
                model(x)
            timings = []
            for _ in range(protocol['measured_calls']):
                started = time.perf_counter_ns()
                prediction = model(x)
                timings.append((time.perf_counter_ns()-started)/1e6)
            if not torch.isfinite(prediction).all():
                raise ValueError('Nonfinite runtime output')
            rows.append(dict(checkpoint_sha256=sha, parameters=sum(p.numel() for p in model.parameters()),
                             checkpoint_bytes=Path(descriptor['path']).stat().st_size,
                             milliseconds=timings, median_ms=float(np.median(timings)),p95_ms=float(np.quantile(timings,.95))))
            print('SEG2 CPU RUNTIME',sha,rows[-1]['median_ms'],flush=True)
    write_once(RUN/'runtime.json',dict(status='complete',pid=os.getpid(),protocol_sha256=digest(RUN/'runtime_protocol.json'),rows=rows,finished_utc=utc()))


def comparisons(selection,results):
    lookup = {(r['checkpoint_sha256'],r['source']):r for r in results['records']}
    if (len(lookup) != len(results['records'])
            or set(lookup) != {(sha,s) for sha in selected_models(selection) for s in SOURCES}):
        raise ValueError('Incomplete or duplicate model/source comparison coverage')
    data_cache = {}
    def arrays(sha,source):
        key = sha,source
        if key not in data_cache:
            # Retain only small per-image fields; packed masks are not needed for comparisons.
            value = arrays_from(lookup[key]['arrays'])
            data_cache[key] = {name:value[name] for name in ('global_ids','confusion','color_error')}
        return data_cache[key]
    rows,contrasts,longer = [],[],[]
    choices = {(c['arm'],c['seed'],c['budget']):c for c in selection['choices']}
    for choice in selection['choices']:
        sha = choice['checkpoint']['sha256']
        sources = {}
        for source in SOURCES:
            metric,base = lookup[sha,source]['metrics'],lookup[INITIAL_SHA,source]['metrics']
            sources[source] = dict(metrics=metric, iou_points_vs_seg1=100*(metric['mean_image_iou']-base['mean_image_iou']),
                                   color_vs_seg1=paired_color(arrays(INITIAL_SHA,source),arrays(sha,source)))
        rows.append(dict(arm=choice['arm'],seed=choice['seed'],budget=choice['budget'],selected_step=choice['step'],
                         checkpoint_sha256=sha,selection_score=choice['selection_score'],sources=sources))
    for seed in SEEDS:
        for budget in BUDGETS:
            a,b = (choices[arm,seed,budget]['checkpoint']['sha256'] for arm in ARMS)
            per_source = {source:dict(iou_points=100*(lookup[b,source]['metrics']['mean_image_iou']-lookup[a,source]['metrics']['mean_image_iou']),
                                     color=paired_color(arrays(a,source),arrays(b,source))) for source in SOURCES}
            contrasts.append(dict(seed=seed,budget=budget,sources=per_source,
                                   macro_iou_points=sum(r['iou_points'] for r in per_source.values())/2))
    for arm in ARMS:
        for seed in SEEDS:
            for lower,upper in zip(BUDGETS[:-1],BUDGETS[1:],strict=True):
                a,b = (choices[arm,seed,budget]['checkpoint']['sha256'] for budget in (lower,upper))
                per_source = {s:100*(lookup[b,s]['metrics']['mean_image_iou']-lookup[a,s]['metrics']['mean_image_iou']) for s in SOURCES}
                longer.append(dict(arm=arm,seed=seed,lower_budget=lower,upper_budget=upper,iou_points=per_source,
                                   macro_iou_points=sum(per_source.values())/2))
    return dict(all_choices=rows,data_contrasts=contrasts,longer_training_contrasts=longer,
                baseline={s:lookup[INITIAL_SHA,s]['metrics'] for s in SOURCES},
                overall_validation_choice=selection['overall'])


def report():
    if (RUN/'verification.json').exists():
        seal = read(RUN/'verification.json')
        if seal.get('passed') is not True:
            raise ValueError('Invalid Seg2 seal')
        check_bindings(seal['artifacts'])
        check_bindings(read(RUN/'registration.json')['bindings'])
        print('SEG2 READ-ONLY VERIFIED',digest(RUN/'verification.json'),flush=True)
        return
    require_predecessors()
    selection,completion,_ = completed_selection()
    audit_value,runtime_value = require_finished(RUN/'test_audit.json'),require_finished(RUN/'runtime.json')
    if audit_value.get('passed') is not True:
        raise ValueError('Held evaluation audit required')
    check_bindings(audit_value['artifacts'])
    checked(RUN/'runtime_protocol.json',runtime_value['protocol_sha256'])
    runtime_protocol = read(RUN/'runtime_protocol.json')
    checked(RUN/'selection.json',runtime_protocol['selection_sha256'])
    checked(RUN/'test_audit.json',runtime_protocol['test_audit_sha256'])
    expected = set(selected_models(selection))
    if len(runtime_value['rows']) != len(expected) or {r['checkpoint_sha256'] for r in runtime_value['rows']} != expected:
        raise ValueError('Missing CPU runtime for a selected checkpoint')
    for row in runtime_value['rows']:
        if (row['parameters'] != PARAMETERS or len(row['milliseconds']) != 60
                or not all(math.isfinite(v) and v>0 for v in row['milliseconds'])
                or row['median_ms'] != float(np.median(row['milliseconds']))
                or row['p95_ms'] != float(np.quantile(row['milliseconds'],.95))):
            raise ValueError('Invalid model size or runtime arithmetic')
    value = comparisons(selection,read(RUN/'test_results.json'))
    value.update(training_seconds=completion['elapsed_seconds'],parameters=PARAMETERS,
                 new_updates=completion['successful_updates'],new_image_presentations=recipe()['new_image_presentations'],
                 runtime=runtime_value['rows'],ordinary_phone_instrument_accuracy=None,
                 evaluation_limit='LaPa test previously exposed; CelebA held within source, cross-source person independence unverified')
    write_once(RUN/'summary.json',value)
    lines = ['# Luma ChromaSeed-Seg2 — результаты сравнения','',
             '4 416 673 параметра; исходные веса Seg1. Шесть дообучений: два состава данных × три начальных значения генератора.',
             'Все модели выбраны по валидации до теста. Ниже показаны все 18 вариантов, включая ухудшения.', '',
             '| Данные | Seed | Бюджет шагов | Выбранный шаг | LaPa IoU, % | CelebA IoU, % |',
             '|---|---:|---:|---:|---:|---:|']
    for row in value['all_choices']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['budget']} | {row['selected_step']} | {100*row['sources']['lapa']['metrics']['mean_image_iou']:.3f} | {100*row['sources']['celeba']['metrics']['mean_image_iou']:.3f} |")
    lines.extend(['','## Добавление данных: смешанный набор минус LaPa','',
                  '| Seed | Бюджет | LaPa, п. п. IoU | CelebA, п. п. IoU | Среднее, п. п. |',
                  '|---:|---:|---:|---:|---:|'])
    for row in value['data_contrasts']:
        lines.append(f"| {row['seed']} | {row['budget']} | {row['sources']['lapa']['iou_points']:+.3f} | {row['sources']['celeba']['iou_points']:+.3f} | {row['macro_iou_points']:+.3f} |")
    lines.extend(['','## Более долгое дообучение','',
                  '| Данные | Seed | Бюджеты | Изменение среднего IoU, п. п. |','|---|---:|---:|---:|'])
    for row in value['longer_training_contrasts']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['lower_budget']} → {row['upper_budget']} | {row['macro_iou_points']:+.3f} |")
    lines.extend(['',f"Общее время обучения и валидации: {value['training_seconds']/60:.2f} мин. Новых обновлений: {value['new_updates']}; предъявлений снимков: {value['new_image_presentations']}.",
                  '', 'Полные ошибки видимого цвета, доли плохих масок, покрытия, все сравнения с Seg1 и измеренные задержки каждого выбранного файла сохранены в `summary.json`.',
                  '', 'IoU оценивает совпадение маски кожи, а не долю фотографий с верным физическим оттенком. Ошибка цвета рассчитана по маскам на одном и том же изображении при предположении sRGB/D65/2°. Это не измеренный цвет кожи.',
                  '', 'LaPa TEST уже использовался раньше. Разделение CelebA проверено внутри источника; отсутствие одинаковых людей между источниками не доказано. Три seed не дают основания для уверенного заявления об обобщении на рынок.',
                  '', 'Время CPU относится к проходу модели на этом компьютере, FP32, два потока, batch 1; декодирование фото и обработка цвета не включены. Скорость на телефоне не измерена.',
                  '', 'Проверка включает целостность файлов, воспроизведение всех потоков обучающих индексов и полный пересчёт метрик из сохранённых масок. Независимое повторное обучение и повторный нейросетевой инференс не проводились.',''])
    with (RUN/'report.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines))
    artifacts = {**audit_value['artifacts'],**read(RUN/'training_audit.json')['artifacts']}
    for name in ('test_audit.json','runtime_protocol.json','runtime.json','summary.json','report.md'):
        artifacts[str(RUN/name)] = digest(RUN/name)
    check_bindings(artifacts)
    write_once(RUN/'verification.json',dict(passed=True,artifacts=artifacts,created_utc=utc(),
                                            classification='verified experiment progress; broader goal remains active'))
    print('SEG2 REPORT VERIFIED',str(RUN/'report.md'),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['audit','runtime','report'])
    args = parser.parse_args()
    {'audit':audit,'runtime':runtime,'report':report}[args.command]()


if __name__ == '__main__':
    main()
