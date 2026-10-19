"""Inner-only pruning transfer diagnostic; never use outer outcomes.

Uses the complete independently audited cohort by default. Smoke mode writes
a separate explicitly incomplete QA artifact. Fold observations overlap,
and inherited source recipes were selected on these same inner folds: the
validation deltas are descriptive development, not unbiased risk estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results/thinned-atomic-classifier'
METHODS = ('proposed', 'unguarded_refined', 'raw_repair_refined',
           'full_refined', 'fgs_refined', 'sfrip_refined')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inner_rows(dataset, dimension, blocks):
    rows = []
    for fold, block in enumerate(blocks):
        for method in METHODS:
            before, after = block['options'][method]
            assert before['variant'] == 0 and after['variant'] == 1
            pid = after['metric']['pruning_id']
            fit = block['pruning_records'][pid]['fit']
            assert before['metric']['features'] == fit['initial_features']
            assert after['metric']['features'] == fit['features']
            training_delta = fit['final_hard_LOO_BA']-fit['initial_hard_LOO_BA']
            validation_delta = (after['evaluation']['metrics']['clean_BA']
                                -before['evaluation']['metrics']['clean_BA'])
            stress_delta=fit['final_thinning_score']['mean_BA']-fit['initial_thinning_score']['mean_BA']
            assert stress_delta>=-1e-12
            assert training_delta >= -1e-12
            rows.append({'dataset': dataset, 'fold': fold, 'method': method,
                'guard_reference': fit['guard_reference'],
                'initial_count': len(fit['initial_features']), 'final_count': len(fit['features']),
                'removed_additional_fraction': (len(fit['initial_features'])-len(fit['features']))/dimension,
                'training_stress_delta_pp':100*stress_delta, 'training_delta_pp': 100*training_delta, 'inner_validation_delta_pp': 100*validation_delta,
                'pruned': bool(fit['chosen_ids']),
                'training_gain_validation_loss': bool(training_delta>1e-12 and validation_delta<-1e-12),
                'training_tie_validation_loss': bool(abs(training_delta)<=1e-12 and validation_delta<-1e-12)})
    return rows


def aggregate(rows):
    summaries = {}
    for method in METHODS:
        data = [r for r in rows if r['method'] == method]
        datasets = sorted({r['dataset'] for r in data})
        by_dataset = []
        for name in datasets:
            block = [r for r in data if r['dataset'] == name]
            by_dataset.append({'dataset': name,
                **{key: float(np.mean([r[key] for r in block])) for key in
                   ('training_stress_delta_pp', 'training_delta_pp', 'inner_validation_delta_pp', 'removed_additional_fraction')},
                'pruned_folds': sum(r['pruned'] for r in block), 'folds': len(block)})
        summaries[method] = {'datasets': len(datasets), 'folds': len(data),
            'pruned_folds': sum(r['pruned'] for r in data),
            'training_gain_validation_loss': sum(r['training_gain_validation_loss'] for r in data),
            'training_tie_validation_loss': sum(r['training_tie_validation_loss'] for r in data),
            'validation_improved_folds': sum(r['inner_validation_delta_pp']>1e-10 for r in data),
            'validation_worsened_folds': sum(r['inner_validation_delta_pp']<-1e-10 for r in data),
            'dataset_equal_means': {key: float(np.mean([r[key] for r in by_dataset])) for key in
                                   ('training_stress_delta_pp', 'training_delta_pp', 'inner_validation_delta_pp', 'removed_additional_fraction')},
            'by_dataset': by_dataset}
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', choices=['iris'])
    args = parser.parse_args()
    if args.smoke:
        names = [args.smoke]; audit_hash = None
    else:
        audit_path = FOLDER/'complete-audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['passed']
        assert sha(FOLDER/'complete.json') == audit['summary_sha256']
        names = sorted(audit['artifact_hashes'])
        assert len(names) == 15
        assert all(sha(FOLDER/(name+'.json')) == audit['artifact_hashes'][name] for name in names)
        audit_hash = sha(audit_path)
    rows, hashes = [], {}
    for name in names:
        path = FOLDER/(name+'.json'); saved = json.loads(path.read_text())
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        # Deliberately pass only inner blocks to analysis; no outer reads.
        rows += inner_rows(name, saved['d'], saved['inner'])
        hashes[name] = sha(path)
    result = {'complete_cohort': not bool(args.smoke), 'smoke_only': bool(args.smoke),
        'scope': 'inner_only_dependent_development_transfer_not_unbiased_generalization_or_superiority',
        'audit_sha256': audit_hash, 'artifact_hashes': hashes, 'rows': rows, 'summary': aggregate(rows),
        'limitations': ['Inner folds overlap; no inferential p-values.',
            'Inherited source recipe selected on same inner validation folds.',
            'Repeated development observations cannot establish population superiority.']}
    suffix = '-smoke' if args.smoke else ''
    out = ROOT/f'research/thinned-atomic-transfer-diagnostic{suffix}.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'artifact_hashes')}, indent=2))


if __name__ == '__main__':
    main()
