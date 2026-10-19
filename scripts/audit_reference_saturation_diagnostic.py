"""Replay all permuted labels with the existing independent FRS implementation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import positive_region,_distance


def main():
    path = ROOT/'results/reference-saturation-diagnostic.json'; saved = json.loads(path.read_text()); counts = {'references':0,'positive_region_replays':0}
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    for row in saved['rows']:
        name = row['dataset']; source_path = ROOT/'results/nested-geometry-forest'/(name+'.json')
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == row['source_sha256']
        source = json.loads(source_path.read_text()); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
        X,y = load_dataset(name); assert row['d'] == X.shape[1] and len(expected) == len(row['records'])
        for record,(i,j,s,inner) in zip(row['records'],expected):
            assert record['outer_index'] == i and record['inner_index'] == j
            assert record['outer_reference_indices'] == s['reference_indices'] and record['reference_indices'] == inner['reference_indices']
            ri,a = np.array(record['outer_reference_indices']),np.array(record['reference_indices']); A,_ = preprocess_fit(X[ri[a]],X[ri[a]]); labels = y[ri[a]]
            assert record['seed'] == 20261015+i*17+j; rng = np.random.default_rng(record['seed'])
            assert record['label_permutations'] == [rng.permutation(len(labels)).tolist() for _ in range(5)]
            for method,weight in (('sqrt_d',1/np.sqrt(A.shape[1])),('unit_mass',1/A.shape[1])):
                full_weights = np.repeat(weight,A.shape[1]); values = [record['methods'][method]['actual']]+record['methods'][method]['shuffled']
                permutations = [np.arange(len(labels))]+[np.array(p) for p in record['label_permutations']]
                for reported,p in zip(values,permutations):
                    assert sorted(p) == list(range(len(labels)))
                    gamma = positive_region(A,labels[p],weights=full_weights)
                    distance = _distance(A,np.arange(A.shape[1]),full_weights)
                    distance[labels[p,None] == labels[p][None,:]] = np.inf; enemies = distance.min(axis=1)
                    np.testing.assert_allclose(reported['gamma'],gamma.mean(),atol=1e-12)
                    np.testing.assert_allclose(reported['saturation_fraction'],np.mean(enemies >= 1),atol=1e-12)
                    np.testing.assert_allclose(reported['minimum_enemy_distance'],enemies.min(),atol=1e-12)
                    np.testing.assert_allclose(reported['median_enemy_distance'],np.median(enemies),atol=1e-12)
                    counts['positive_region_replays'] += 1
                np.testing.assert_allclose(record['methods'][method]['label_discrimination_gap'],values[0]['gamma']-np.mean([v['gamma'] for v in values[1:]]),atol=1e-12)
            counts['references'] += 1
        for method in ('sqrt_d','unit_mass'):
            actual = [r['methods'][method]['actual'] for r in row['records']]; shuffled = [v for r in row['records'] for v in r['methods'][method]['shuffled']]
            metrics = row['metrics'][method]
            for field,expected_value in (('gamma',np.mean([v['gamma'] for v in actual])),('saturation',np.mean([v['saturation_fraction'] for v in actual])),
                ('shuffled_gamma',np.mean([v['gamma'] for v in shuffled])),('label_gap',np.mean([r['methods'][method]['label_discrimination_gap'] for r in row['records']]))):
                np.testing.assert_allclose(metrics[field],expected_value,atol=1e-12)
    assert counts['references'] == 128 and counts['positive_region_replays'] == 1536
    result = {'passed':True,'counts':counts,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'train_only_permutation_and_existing_positive_region_implementation_replay',
        'limits':'Nominal empirical diagnostic only; normalization changes relation; no predictive superiority claim'}
    (ROOT/'results/reference-saturation-diagnostic-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
