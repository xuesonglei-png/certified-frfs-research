"""Replay training-only classifier selection and all paired budget predictions."""
import argparse
import hashlib
import json

from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_common_fuzzy_classifier import train, score


def main(source_name):
    threadpool_limits(1)
    folder = ROOT / 'results' / (source_name + '-common-budget')
    summary = json.loads((folder / 'complete.json').read_text())
    counts = {'outer_splits': 0, 'inner_fits': 0, 'outer_fits': 0, 'count_matches': 0}
    for row in summary['rows']:
        name = row['dataset']
        p = json.loads((folder / (name + '.json')).read_text())
        source = json.loads((ROOT / 'results' / source_name / (name + '.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in p['fingerprints'].items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
        for index, (split, original) in enumerate(zip(p['splits'], source['splits'])):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            assert split['reference_indices'] == original['reference_indices']
            assert split['validation_indices'] == original['validation_indices']
            raw, labels = X[ri], y[ri]
            for method, choices in split['trace'].items():
                for option in choices:
                    values, sizes = [], []
                    for fold, inner in zip(option['folds'], original['inner']):
                        a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                        assert set(a).isdisjoint(b)
                        A, B = preprocess_fit(raw[a], raw[b])
                        target = inner['by_C'][str(original['selected_config']['C'])]['stage']['policies'][original['chosen']]['candidate']['features']
                        assert len(fold['features']) == len(target)
                        pred_score = score(B, labels[b], train(A, labels[a], fold['features'], option['C']))
                        np.testing.assert_allclose(pred_score, fold['clean'], atol=1e-12)
                        values.append(pred_score)
                        sizes.append(len(fold['features']))
                        counts['inner_fits'] += 1
                    np.testing.assert_allclose(np.mean(values), option['mean_clean'], atol=1e-12)
                    np.testing.assert_allclose(np.mean(sizes), option['mean_size'], atol=1e-12)
                selected = max(range(len(choices)), key=lambda j: (choices[j]['mean_clean'], -choices[j]['mean_size'], -j))
                record = split['results'][method]
                assert record['chosen_index'] == selected
                assert len(record['features']) == split['target_size']
                A, B = preprocess_fit(raw, X[vi])
                fit = train(A, labels, record['features'], choices[selected]['C'])
                np.testing.assert_allclose(fit['coef'], record['fit']['coef'], atol=1e-12)
                np.testing.assert_allclose(fit['intercept'], record['fit']['intercept'], atol=1e-12)
                rng = np.random.default_rng(144+index)
                noisy = [score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), y[vi], fit) for _ in range(3)]
                np.testing.assert_allclose(score(B, y[vi], fit), record['metrics']['clean'], atol=1e-12)
                np.testing.assert_allclose(np.mean(noisy), record['metrics']['query_noise01'], atol=1e-12)
                counts['outer_fits'] += 1
                counts['count_matches'] += 1
            counts['outer_splits'] += 1
        print(name, 'common budget classifier replay passed', flush=True)
    assert counts['outer_splits'] == 44
    counts['scope'] = 'classifier_selection_prediction_and_budget_replay_not_native_algorithm_fidelity_audit'
    (folder / 'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='nested-triangular-repair')
    main(parser.parse_args().source)
