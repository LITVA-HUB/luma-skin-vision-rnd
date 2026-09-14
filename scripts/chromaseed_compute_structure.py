"""Count dense linear work in the frozen AS/HR forward graph using meta tensors.

No numerical prediction, training, checkpoint loading, GPU context or latency
measurement is performed. Prefix costs below are arithmetic counterfactuals;
the original forward still executes every registered pass.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/Luma-RnD/chromaseed_compute_structure_v1')
P3 = Path('D:/Luma-RnD/chromaseed_palette_transfer_v1/registration.json')
P3_SHA = '90a9a78bc46f5ca1f40befe0cd04e8861374868d41710c099c7573da7a2add17'
ORIGINALS = ('scripts/chromaseed_architecture_scale.py', 'scripts/chromaseed_head_range.py',
             'scripts/chromaseed_refine.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


class LinearTrace:
    """Local observer on a scratch meta-model instance; original code is unchanged."""

    def __init__(self, name, original, inputs, outputs, records):
        self.name, self.original = name, original
        self.inputs, self.outputs, self.records = inputs, outputs, records

    def __call__(self, value):
        if value.device.type != 'meta' or value.shape[-1] != self.inputs:
            raise ValueError('operation audit accepts only matching meta tensors')
        self.records.append(dict(layer=self.name, input_shape=list(value.shape),
                                 macs=math.prod(value.shape[:-1])*self.inputs*self.outputs))
        return self.original(value)


def summarize_trace(records, operator_flops, *, recurrent, instances):
    if (not records or type(instances) is not int or instances < 1
            or set(operator_flops) - {'aten.bmm', 'aten.mm', 'aten.addmm'}):
        raise ValueError('unsupported workload or operator counter')
    total = sum(row['macs'] for row in records)
    if total*2 != sum(operator_flops.values()):
        raise ValueError('linear trace disagrees with independent operator counter')
    if any(row['macs'] % instances for row in records):
        raise ValueError('linear work is not divisible by the number of instances')
    layers = Counter()
    for row in records:
        layers[row['layer']] += row['macs']//instances
    if recurrent:
        if ([row['layer'] for row in records[:4]] != ['token1', 'token2', 'context', 'key']
                or [row['layer'] for row in records[4:]] != ['query', 'update1', 'update2', 'head']*4):
            raise ValueError('unexpected recurrent call order')
        fixed = sum(row['macs'] for row in records[:4])//instances
        passes = [sum(row['macs'] for row in records[4+4*i:8+4*i])//instances for i in range(4)]
        prefixes = []
        for count in range(1, 5):
            work = fixed + sum(passes[:count])
            prefixes.append(dict(passes=count, linear_macs=work,
                                 linear_mac_reduction=1-work/(total//instances),
                                 is_executed_early_exit=False))
    else:
        fixed, passes = total//instances, []
        prefixes = []
    return dict(bank_linear_macs=total, linear_macs=total//instances,
                by_layer_linear_macs=dict(layers), fixed_linear_macs=fixed,
                per_pass_linear_macs=passes, passes=4 if recurrent else 1,
                prefix_counterfactuals=prefixes)


def audit_variant(variant, mode, *, slots=1, batch=1):
    if any(type(value) is not int or value < 1 for value in (slots, batch)):
        raise ValueError('slots and batch must be positive integers')
    import chromaseed_head_range as hr
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    if torch.cuda.is_initialized():
        raise RuntimeError('run shape analysis in a process without a CUDA context')
    records = []
    with torch.device('meta'):
        net = hr.Bank(variant, mode, seeds=tuple(17+i for i in range(slots))).eval()
        if any(parameter.device.type != 'meta' for parameter in net.parameters()):
            raise RuntimeError('unexpected real model allocation')
        for name, inputs, outputs in hr.original.specs(variant):
            net.layers[name] = LinearTrace(name, net.layers[name], inputs, outputs, records)
        with FlopCounterMode(display=False) as counter:
            output, _ = net(torch.zeros(slots, batch, 36), torch.zeros(slots, batch, 64, 18),
                            torch.zeros(slots, batch, 3))
    ops = {str(op): value for op, value in counter.flop_counts['Global'].items()}
    recurrent = variant.startswith(('soft', 'dynamic'))
    summary = summarize_trace(records, ops, recurrent=recurrent, instances=slots*batch)
    if list(output.shape) != [slots, batch, summary['passes'], 3] or output.device.type != 'meta':
        raise ValueError('unexpected model result shape')
    if torch.cuda.is_initialized():
        raise RuntimeError('shape analysis unexpectedly initialized CUDA')
    return dict(variant=variant, head_mode=mode, slots=slots, batch=batch,
                parameters_per_model_including_anchor=net.theta.numel()//slots+643,
                output_shape=list(output.shape), trace=records, operator_flops=ops, **summary,
                cuda_initialized=False, latency_seconds=None, quality_metric=None)


def prepare():
    if digest(P3) != P3_SHA:
        raise ValueError('reference registration changed')
    registered = json.loads(P3.read_text(encoding='utf-8-sig'))['bindings']
    for name in ORIGINALS:
        if digest(ROOT/name) != registered[name]:
            raise ValueError('frozen source changed: ' + name)
    import chromaseed_head_range as hr
    import torch
    import torch.utils.flop_counter as flop_module

    paths = [ROOT/name for name in ORIGINALS] + [P3, Path(__file__),
             ROOT/'tests/test_chromaseed_compute_structure.py', Path(inspect.getfile(flop_module))]
    inputs = [binding(path) for path in paths]
    single = [audit_variant(variant, mode) for variant in hr.original.VARIANTS for mode in hr.MODES]
    banks = [audit_variant(variant, mode, slots=6, batch=64)
             for variant in hr.original.VARIANTS for mode in hr.MODES]
    for one, bank in zip(single, banks, strict=True):
        if (one['linear_macs'] != bank['linear_macs']
                or bank['bank_linear_macs'] != 384*one['bank_linear_macs']
                or one['parameters_per_model_including_anchor'] != bank['parameters_per_model_including_anchor']):
            raise ValueError('workload scaling mismatch')
    for variant in hr.original.VARIANTS:
        rows = [row for row in single if row['variant'] == variant]
        if len({row['linear_macs'] for row in rows}) != 1:
            raise ValueError('unexpected dense-linear work change across head modes')
    for soft_name, dynamic_name in [('soft_small', 'dynamic_small'), ('soft5m', 'dynamic5m')]:
        soft, dynamic = [[row for row in single if row['variant'] == name and row['head_mode'] == 'unit']
                         for name in (soft_name, dynamic_name)]
        if soft[0]['trace'] != dynamic[0]['trace']:
            raise ValueError('soft/dynamic linear traces unexpectedly differ')
    for item in inputs:
        if digest(item['path']) != item['sha256']:
            raise ValueError('source changed during analysis')
    return dict(schema='luma.chromaseed.compute-structure.v1', inputs=inputs,
                versions=dict(torch=torch.__version__, python=platform.python_version()),
                cases=len(single)+len(banks), single_model_cases=single, bank_cases=banks,
                scope='Dense linear multiply-accumulate counts inside Bank.forward only; one MAC counted as two FLOPs.',
                parameter_scope='Per-model parameters include frozen 643-parameter anchor; anchor execution is outside this forward trace.',
                excluded_work=['anchor prediction', 'normalization', 'nonlinearities', 'attention elementwise scoring and pooling',
                               'gating/top-k/reductions', 'memory traffic', 'backward/optimizer', 'image decoding/segmentation'],
                limitations=['No learned weights or dataset examples were used.',
                             'Counts do not establish latency, training throughput or quality.',
                             'Prefix costs are arithmetic projections, not an implemented or calibrated early-exit policy.',
                             'A dense mask changes connected values; the current forward trace does not skip linear layers.'],
                documentation='https://docs.pytorch.org/docs/2.8/meta.html',
                gpu_context_initialized=False, latency_seconds=None, quality_metric=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify'])
    args = parser.parse_args()
    path = OUT/'analysis.json'
    if args.command == 'run' and path.exists():
        raise FileExistsError('analysis already exists; use verify')
    result = prepare()
    if args.command == 'run':
        OUT.mkdir(parents=True, exist_ok=True)
        result['created_utc'] = datetime.now(timezone.utc).isoformat()
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        saved = json.loads(path.read_text(encoding='utf-8-sig'))
        saved.pop('created_utc')
        if saved != result:
            raise ValueError('operation analysis does not reproduce')
    print(json.dumps(dict(command=args.command, sha256=digest(path), cases=result['cases'],
                          cuda_initialized=result['gpu_context_initialized'],
                          rows=[dict(variant=row['variant'], parameters=row['parameters_per_model_including_anchor'],
                                     dense_linear_macs=row['linear_macs'])
                                for row in result['single_model_cases'] if row['head_mode'] == 'unit'])))


if __name__ == '__main__':
    main()
