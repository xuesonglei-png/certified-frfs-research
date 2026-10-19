"""Replay matched-control margins, support provenance and held-out metrics."""
import argparse
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from compare_matched_robust_controls import ROOT, metrics
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import box_class_margins

def main(folder):
    threadpool_limits(1)
    directory = ROOT / 'results' / folder
    summary = json.loads((directory / 'complete.json').read_text())
    counts = {'splits':0,'models':0,'infeasible_supports':0,'predictions':0}
    gains = []
    for row in summary['rows']:
        name = row['dataset']; p = json.loads((directory / f'{name}.json').read_text())
        t = json.loads((ROOT / 'results/linear-retention' / f'{name}.json').read_text())
        nested = json.loads((ROOT / 'results/nested-classifier-pipelines' / f'{name}.json').read_text())
        for rel, digest in p['fingerprints'].items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
        dscores = []
        for i,s in enumerate(p['splits']):
            teacher = t['splits'][i]; ri = np.array(s['reference_indices']); vi = np.array(s['validation_indices'])
            assert ri.tolist()==teacher['reference_indices']==nested['splits'][i]['reference_indices']
            assert vi.tolist()==teacher['validation_indices']==nested['splits'][i]['validation_indices']
            assert set(ri).isdisjoint(vi)
            X,y = load_dataset(name); A,B = preprocess_fit(X[ri],X[vi]); classes = np.array(teacher['teacher_classes'])
            original = box_class_margins(A,y[ri],np.array(teacher['teacher_coef']),np.array(teacher['teacher_intercept']),classes)
            protected = original > 1e-10; rng = np.random.default_rng(144+i)
            queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            for method,result in s['results'].items():
                fs = result['features']; fit = result['fit']
                if method in ('fgs2023','sfrip2025'):
                    assert fs == nested['splits'][i]['results']['logistic'][method]['features']
                    assert not result['fallback']
                if method == 'robust_full': assert fs == list(range(p['d']))
                if not fit['feasible']:
                    assert result['metrics'] is None
                    counts['infeasible_supports'] += 1
                    continue
                W,b = np.array(fit['coef']),np.array(fit['intercept'])
                assert np.all(W[:,np.setdiff1d(np.arange(p['d']),fs)] == 0)
                margin = box_class_margins(A,y[ri],W,b,classes)
                assert np.all(margin[protected] >= .5*original[protected]-1e-8)
                assert np.all(margin[protected] > 1e-10)
                if 'deployed_margins' in fit:
                    np.testing.assert_allclose(margin,fit['deployed_margins'],atol=1e-12)
                replay = metrics(queries,y[vi],classes,fit)
                for key in ('clean','query_noise01'):
                    np.testing.assert_allclose(replay[key],result['metrics'][key],atol=1e-12)
                assert replay['class_recalls'] == result['metrics']['class_recalls']
                counts['models'] += 1; counts['predictions'] += 4
            dscores.append(s['results']['proposed']['metrics']['clean']-s['results']['robust_full']['metrics']['clean'])
            counts['splits'] += 1
        gains.append(float(np.mean(dscores)))
    np.testing.assert_allclose(np.mean(gains),summary['gain_vs_robust_full'],atol=1e-12)
    assert counts['splits'] == 44
    counts['scope'] = 'margin_support_fingerprint_prediction_replay_not_optimizer_reproduction'
    (directory / 'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(counts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('folder'); main(parser.parse_args().folder)
