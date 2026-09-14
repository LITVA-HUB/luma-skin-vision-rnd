"""Fixed Seg2 comparison, validation-only choices and immutable experiment bindings."""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from skin_face_transfer_data import DATA_ROOT, SOURCES, checked, digest, read

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('D:/Luma-RnD/skin_face_transfer_v1')
SEG1 = DATA_ROOT / 'facial_skin_v1'
INITIAL = SEG1 / 'best.pt'
INITIAL_SHA = '1203cbc5ed2ee17cb2a408c47a23b3a28468f169d48b4d3cee8bd3059e7fbed3'
ARMS = ('lapa_only', 'lapa_celeba')
SEEDS = (17, 29, 43)
BUDGETS = (1494, 2988, 5976)
INTERVAL, BATCH, WIDTH, SIZE, PARAMETERS = 498, 32, 24, 192, 4416673
COUNTS = {'lapa': {'train': 15914, 'validation': 1692, 'test': 2000},
          'celeba': {'train': 24112, 'validation': 2992, 'test': 2822}}
FILES = (
    'scripts/skin_face_transfer_data.py',
    'scripts/skin_face_transfer_study.py',
    'scripts/skin_face_transfer_run.py',
    'scripts/skin_face_transfer_evaluate.py',
    'scripts/skin_face_transfer_report.py',
    'tests/test_skin_face_transfer_data.py',
    'tests/test_skin_face_transfer_study.py',
    'tests/test_skin_face_transfer_run.py',
    'tests/test_skin_face_transfer_evaluate.py',
    'docs/superpowers/specs/2026-09-14-face-transfer-design.md',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)


def check_bindings(bindings):
    for name, sha in bindings.items():
        path = Path(name)
        checked(path if path.is_absolute() else ROOT / path, sha)


def recipe():
    trajectories = [dict(id=f'{arm}_s{seed}', arm=arm, seed=seed, steps=BUDGETS[-1],
                         image_presentations=BUDGETS[-1]*BATCH,
                         source_presentations={'lapa': BUDGETS[-1]*BATCH if arm == ARMS[0] else BUDGETS[-1]*BATCH//2,
                                               'celeba': 0 if arm == ARMS[0] else BUDGETS[-1]*BATCH//2})
                    for arm in ARMS for seed in SEEDS]
    return dict(name='Luma ChromaSeed-Seg2', arms=list(ARMS), seeds=list(SEEDS),
                trajectories=trajectories, budgets=list(BUDGETS), validation_interval=INTERVAL,
                validation_steps=list(range(0, BUDGETS[-1]+1, INTERVAL)),
                batch_size=BATCH, parameters=PARAMETERS, width=WIDTH, input_size=SIZE,
                initial_sha256=INITIAL_SHA, optimizer='fresh AdamW', learning_rate=.0001,
                weight_decay=.0001, gradient_clip_norm=5.,
                schedule='cosine multiplier 1 to .1 over 5976 updates; identical prefixes',
                sampler='16 shared LaPa anchors + 16 independent source samples; complete shuffled cycles',
                augmentation='frozen Seg1 augment; equal torch RNG seed per paired trajectory',
                selection='equal source mean-image IoU; maximum within prefix; earliest step on ties',
                counts=copy.deepcopy(COUNTS), new_updates=len(trajectories)*BUDGETS[-1],
                new_image_presentations=len(trajectories)*BUDGETS[-1]*BATCH,
                test_accessed=False, measured_color_targets=False,
                cross_source_person_independence_verified=False,
                prior_lapa_test_exposed=True, cuda_precision='FP16 autocast; FP32 saved weights',
                failed_gradient_policy='fail immediately; no skipped optimizer update counted as success')


def validation_score(record):
    if 'test' in record or 'test_results' in record:
        raise ValueError('TEST results must not enter checkpoint selection')
    if set(record.get('validation', {})) != set(SOURCES):
        raise ValueError('Both validation sources are required')
    values = []
    for source in SOURCES:
        metric = record['validation'][source]
        value = metric.get('mean_image_iou')
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 <= value <= 1
                or metric.get('images') != COUNTS[source]['validation']):
            raise ValueError('Invalid validation metric or source count')
        values.append(value)
    return math.fsum(values)/len(SOURCES)


def select_prefix(history, budget):
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0 or not history:
        raise ValueError('Nonnegative budget and nonempty validation history required')
    previous, eligible = -1, []
    for record in history:
        step = record.get('step')
        if (not isinstance(step, int) or isinstance(step, bool) or step <= previous
                or (previous == -1 and step != 0)):
            raise ValueError('History must start at step zero and increase strictly')
        score = validation_score(record)
        checkpoint = record.get('checkpoint', {})
        if (not isinstance(checkpoint.get('path'), str) or not checkpoint['path']
                or not isinstance(checkpoint.get('sha256'), str) or len(checkpoint['sha256']) != 64):
            raise ValueError('Checkpoint identity required')
        if step <= budget:
            eligible.append((score, step, record))
        previous = step
    score, _, selected = min(eligible, key=lambda r: (-r[0], r[1]))
    value = copy.deepcopy(selected)
    value.update(budget=budget, selection_score=score)
    return value


def freeze_choices(histories):
    spec = recipe()
    if set(histories) != {r['id'] for r in spec['trajectories']}:
        raise ValueError('All six complete trajectories are required')
    choices = []
    initial = set()
    for trajectory in spec['trajectories']:
        history = histories[trajectory['id']]
        if [r.get('step') for r in history] != spec['validation_steps']:
            raise ValueError('Every complete validation checkpoint is required')
        for budget in BUDGETS:
            selected = select_prefix(history, budget)
            selected.update(trajectory_id=trajectory['id'], arm=trajectory['arm'], seed=trajectory['seed'])
            choices.append(selected)
        initial.add(history[0]['checkpoint']['sha256'])
    if len(initial) != 1:
        raise ValueError('Trajectories have different step-zero initializations')
    full = [c for c in choices if c['budget'] == BUDGETS[-1]]
    overall = min(full, key=lambda c: (-c['selection_score'], c['step'], ARMS.index(c['arm']), SEEDS.index(c['seed'])))
    return dict(choices=choices, overall=copy.deepcopy(overall), test_accessed=False,
                choice_scope='validation only; all six trajectories and three prefix budgets')


def initial_bindings():
    pins = {'best.pt': INITIAL_SHA,
            'protocol.json': '42d917ced2431a70b29d1a21041d40070f5b3ad87e3d13df1cf6b023b4b97083',
            'selection.json': '4eece641e844a14aeadfdb79544976b2f4661fedf4efe9e49bd5c724f6e92627',
            'final_verification.json': '3231d667a9091e05678fa59e3b1ba0f825ce2ea5f1d503df12465509d6dd01e3'}
    bindings = {str(SEG1 / name): sha for name, sha in pins.items()}
    check_bindings(bindings)
    seal = read(SEG1 / 'final_verification.json')
    if seal['status'] != 'verified' or seal['artifacts']['best.pt'] != INITIAL_SHA:
        raise ValueError('Invalid Seg1 initialization seal')
    bindings.update({str(SEG1 / name): sha for name, sha in seal['artifacts'].items()})
    for filename in ('protocol.json', 'appearance_protocol.json', 'runtime_protocol.json'):
        protocol = read(SEG1 / filename)
        bindings.update(protocol.get('sources', {}))
    bindings['scripts/skin_face_verify.py'] = seal['source_sha256']
    check_bindings(bindings)
    return bindings


def data_bindings():
    lapa = DATA_ROOT / 'lapa'
    celeba = DATA_ROOT / 'celeba_mask_hq'
    pins = {str(lapa / 'prepared_192/profile.json'): '67e7084af989dcc9de8635d39a0a73949963a9264c147941da0b279dcfbe5625',
            str(celeba / 'prepared_192_v1/profile.json'): 'bf8bd9b233a89077aca4d4e715c5dc15445e8a4c5bb119b83669f7dc185e5d1a',
            str(celeba / 'split_v1/profile.json'): '9e172e2648269bdf95bc59ea43230e34ec645d3237222aab08e20aa5e13b5373',
            str(celeba / 'preparation_verification.json'): '6634e3777252305801b23c12408b5adc670be9ca449839df297c61373c8239fc'}
    check_bindings(pins)
    lp = read(lapa / 'prepared_192/profile.json')
    pins[str(lapa / 'usable_triplets.json')] = lp['triplets_sha256']
    for role, record in lp['splits'].items():
        for kind in ('rgb', 'labels'):
            pins[str(lapa / 'prepared_192' / f'{role}_{kind}.npy')] = record[kind+'_sha256']
        name = lapa / 'prepared_192' / f'{role}_order.json'
        pins[str(name)] = digest(name)
    for directory in ('prepared_192_v1', 'split_v1'):
        profile = read(celeba / directory / 'profile.json')
        if directory == 'split_v1':
            pins[str(celeba / directory / 'protocol.json')] = profile['protocol_sha256']
        for filename, record in profile['files'].items():
            pins[str(celeba / directory / filename)] = record['sha256']
    check_bindings(pins)
    return pins


def current_bindings():
    bindings = {**initial_bindings(), **data_bindings()}
    bindings.update({name: digest(ROOT / name) for name in FILES})
    # The gate verifies these existing sealed workflows through their original consumers.
    for name in ('scripts/chromaseed_head_range_verification.py',
                 'scripts/chromaseed_head_range_report.py',
                 'scripts/chromaseed_palette_transfer_report.py',
                 'scripts/chromaseed_palette_transfer_verification.py',
                 'scripts/skin_face_segment.py', 'scripts/skin_face_appearance.py',
                 'src/luma_skin_vision/color.py'):
        actual = digest(ROOT / name)
        if name in bindings and bindings[name] != actual:
            raise ValueError(f'Conflicting source binding: {name}')
        bindings[name] = actual
    return bindings


def freeze_registration():
    if (RUN / 'registration.json').exists():
        raise FileExistsError('Seg2 registration already exists; preserve the registered experiment')
    value = dict(recipe=recipe(), bindings=current_bindings(), created_utc=utc(),
                 ordinary_phone_instrument_accuracy=None,
                 predecessor_order=['HR full verification', 'P3 full verification', 'Seg2 CUDA'])
    write_once(RUN / 'registration.json', value)
    return value


def verify_registration():
    value = read(RUN / 'registration.json')
    if value['recipe'] != recipe():
        raise ValueError('Seg2 recipe changed after registration')
    check_bindings(value['bindings'])
    return value
