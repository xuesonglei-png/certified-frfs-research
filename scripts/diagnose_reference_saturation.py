"""Train-only nominal dependency/shuffled-label diagnostic, no classifiers."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def values(distance,labels,scale):
    cross = labels[:,None] != labels[None,:]
    enemy = np.where(cross,distance*scale,np.inf).min(axis=1)
    membership = np.minimum(1.,enemy)
    return {'gamma':float(membership.mean()),'saturation_fraction':float(np.mean(enemy >= 1)),
        'minimum_enemy_distance':float(enemy.min()),'median_enemy_distance':float(np.median(enemy))}


def main():
    source = ROOT/'results/nested-geometry-forest'; names = [r['dataset'] for r in json.loads((source/'complete.json').read_text())['rows']]
    assert json.loads((source/'complete-audit.json').read_text())['passed']
    rows = []; total = 0
    for name in names:
        source_path = source/(name+'.json'); saved = json.loads(source_path.read_text()); X,y = load_dataset(name); records = []
        verified = json.loads((source/(name+'-audit.json')).read_text())
        assert verified['artifact_sha256'] == hashlib.sha256(source_path.read_bytes()).hexdigest()
        for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
        for i,s in enumerate(saved['splits']):
            ri = np.array(s['reference_indices'])
            for j,inner in enumerate(s['inner']):
                a = np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; A,_ = preprocess_fit(raw,raw)
                distance = cdist(A,A,metric='cityblock'); rng = np.random.default_rng(20261015+i*17+j)
                permutations = [rng.permutation(len(labels)) for _ in range(5)]
                methods = {}
                for method,scale in (('sqrt_d',1/np.sqrt(A.shape[1])),('unit_mass',1/A.shape[1])):
                    actual = values(distance,labels,scale); shuffled = [values(distance,labels[p],scale) for p in permutations]
                    methods[method] = {'actual':actual,'shuffled':shuffled,
                        'label_discrimination_gap':actual['gamma']-float(np.mean([r['gamma'] for r in shuffled]))}
                records.append({'outer_index':i,'inner_index':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
                    'seed':20261015+i*17+j,'label_permutations':[p.tolist() for p in permutations],'methods':methods})
        total += len(records)
        metrics = {m:{
            'gamma':float(np.mean([r['methods'][m]['actual']['gamma'] for r in records])),
            'saturation':float(np.mean([r['methods'][m]['actual']['saturation_fraction'] for r in records])),
            'shuffled_gamma':float(np.mean([v['gamma'] for r in records for v in r['methods'][m]['shuffled']])),
            'label_gap':float(np.mean([r['methods'][m]['label_discrimination_gap'] for r in records]))} for m in ('sqrt_d','unit_mass')}
        payload = {'dataset':name,'d':X.shape[1],'records':records,'metrics':metrics,
            'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()}
        rows.append(payload)
    assert total == 128 and len(rows) == 15
    result = {'rows':rows,'records':total,'scope':'train_only_nominal_clipped_full_reference_saturation_5_label_permutations_no_classifiers',
        'limits':'Different_normalization_changes_reference_relation; empirical_label_gap_not_population_or_predictive_advantage; no_query_labels_used',
        'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'research/reference-saturation-diagnostic-protocol.md',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'src/certified_frfs/datasets.py')}}
    (ROOT/'results/reference-saturation-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines = ['# Train-only full fuzzy-reference saturation', '',
        'All15/all128 archived search references; five class-count-preserving label permutations per reference. No classifier fitting or query-label access.', '',
        '|Dataset|d|Original gamma|Original saturated %|Shuffled gamma|Original label gap|Unit-mass gamma|Unit-mass label gap|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        a,b = row['metrics']['sqrt_d'],row['metrics']['unit_mass']
        lines.append(f"|{row['dataset']}|{row['d']}|{a['gamma']:.5f}|{100*a['saturation']:.2f}|{a['shuffled_gamma']:.5f}|{a['label_gap']:.5f}|{b['gamma']:.5f}|{b['label_gap']:.5f}|")
    lines += ['', 'Changing to unit-mass weights changes the full fuzzy relation; it is not a tighter certificate for the old relation. High gamma alone cannot imply good test classification. An empirical saturation observation does not invalidate the original sound upper bound or establish general predictive superiority.', '',
        'These diagnostics may motivate a separately frozen normalized-reference method study. Do not select normalization using these query scores: none were accessed. Any subsequent classification result remains a new development experiment requiring exact-count controls, independent audit and untouched confirmation.']
    (ROOT/'research/reference-saturation-diagnostic.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print([{k:v for k,v in row.items() if k in ('dataset','d','metrics')} for row in rows],flush=True)


if __name__ == '__main__': main()
