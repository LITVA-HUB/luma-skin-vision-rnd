"""Use author partitions and person codes; remove transitive duplicate leakage."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
POOL = ROOT / 'prepared_192_v1'
OUT = ROOT / 'split_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def source_table(filename):
    path = ROOT / filename
    receipt = json.loads(path.with_suffix(path.suffix+'.source.json').read_text())
    assert sha(path) == receipt['sha256']
    pairs = [line.split() for line in path.read_text().splitlines()]
    result = {name:int(value) for name, value in pairs}
    assert len(result) == len(pairs) == 202599
    assert set(result) == {f'{i:06d}.jpg' for i in range(1, 202600)}
    return result


def main():
    partition = source_table('list_eval_partition.txt')
    person = source_table('identity_CelebA.txt')
    assert set(partition.values()) == {0, 1, 2}
    assert len(set(person.values())) == 10177
    profile = json.loads((POOL/'profile.json').read_text())
    assert sha(POOL/'rows.json') == profile['files']['rows.json']['sha256']
    rows = json.loads((POOL/'rows.json').read_text())
    assert len(rows) == 30000 and all(r['index'] == i for i,r in enumerate(rows))
    OUT.mkdir(exist_ok=False)
    contract = dict(source_bindings={str(p):sha(p) for p in
                        [POOL/'profile.json', POOL/'rows.json', ROOT/'list_eval_partition.txt', ROOT/'identity_CelebA.txt']},
                    script_sha256=sha(__file__),
                    selection='Union rows by author person code OR exact original JPEG SHA256, transitively. '
                              'Preserve test over validation over train: exclude rows in lower-priority partitions '
                              'within a connected group. Within the surviving partition keep lowest image index '
                              'for each identical JPEG. Do not reassign author partitions.',
                    role='local auxiliary face-mask experiment',
                    identity_use='Provided anonymous source codes only for grouping; never inferred from appearance or fed to model.',
                    numeric_model_results_read=False, instrument_color_targets=False,
                    no_new_independent_phone_quality_claim=True, registered_HR_P3_unchanged=True)
    write_json(OUT/'protocol.json', contract)
    parent = list(range(len(rows)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        x,y = find(a),find(b)
        if x != y:
            parent[max(x,y)] = min(x,y)
    seen_people, seen_bytes = {}, {}
    for r in rows:
        i,p = r['index'],person[r['original_file']]
        for seen,key in [(seen_people,p),(seen_bytes,r['source_sha256'])]:
            if key in seen:
                union(i,seen[key])
            else:
                seen[key] = i
    members = collections.defaultdict(list)
    for r in rows:
        members[find(r['index'])].append(r)
    group_priority = {g:max(partition[r['original_file']] for r in rr) for g,rr in members.items()}
    retained_hashes, decisions = set(), []
    labels = ['train','validation','test']
    for r in rows:
        i = r['index'];g = find(i);code = partition[r['original_file']]
        reason = None
        if code < group_priority[g]:
            reason = 'lower_priority_person_or_exact_image_component'
        elif r['source_sha256'] in retained_hashes:
            reason = 'exact_image_duplicate_in_surviving_partition'
        else:
            retained_hashes.add(r['source_sha256'])
        decisions.append(dict(index=i, author_partition=labels[code],
                              person_code=person[r['original_file']], conservative_group=g,
                              original_file=r['original_file'], source_sha256=r['source_sha256'],
                              role='excluded' if reason else labels[code], reason=reason))
    kept = [r for r in decisions if r['role'] != 'excluded']
    groups, people, hashes = {}, {}, {}
    counts = {}
    for role in labels:
        rr = [r for r in kept if r['role'] == role]
        ids = np.array([r['index'] for r in rr],dtype=np.int32)
        np.save(OUT/(role+'_indices.npy'),ids,allow_pickle=False)
        groups[role] = {r['conservative_group'] for r in rr}
        people[role] = {r['person_code'] for r in rr}
        hashes[role] = {r['source_sha256'] for r in rr}
        counts[role] = dict(images=len(rr), provided_person_codes=len(people[role]),
                           conservative_groups=len(groups[role]))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not groups[a] & groups[b]
        assert not people[a] & people[b]
        assert not hashes[a] & hashes[b]
    assert len({r['source_sha256'] for r in kept}) == len(kept)
    write_json(OUT/'rows.json', decisions)
    summary = dict(author_image_counts=dict(collections.Counter(r['author_partition'] for r in decisions)),
                   kept=counts, retained_images=len(kept), excluded_images=len(rows)-len(kept),
                   exclusions=dict(collections.Counter(r['reason'] for r in decisions if r['reason'])),
                   source_person_codes=len(seen_people),
                   person_component_and_exact_image_sets_disjoint=True,
                   labels='Binary facial skin segmentation only; measured skin-color targets absent.',
                   duplicate_limit='No exact JPEG overlap with LaPa; cross-source reencodings, crops, or person aliases remain unverified.',
                   pool_profile_sha256=sha(POOL/'profile.json'),
                   protocol_sha256=sha(OUT/'protocol.json'),
                   files={name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name)) for name in
                          ['rows.json','train_indices.npy','validation_indices.npy','test_indices.npy']})
    write_json(OUT/'profile.json',summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
