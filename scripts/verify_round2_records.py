"""Cross-check retained research records without rerunning optimization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {'scope': 'Saved-record consistency and provenance, not a new experiment or formal proof', 'checks': {}}
    checks = result['checks']
    old = read('output/bundle-manifest.json')
    assert sha(ROOT/old['bundle']) == old['sha256']
    checks['first_cycle_zip_unchanged'] = True
    for name in ['relative-pilot', 'normalized-pilot', 'normalized-prototypes']:
        saved = read(f'results/round2/{name}.json')
        assert saved['source_files_unchanged_during_run']
        for relative, expected in saved['source_hashes'].items():
            assert sha(ROOT/relative) == expected, (name, relative)
        checks[name+'_recorded_sources_match_current_files'] = True
    for name in ['relative-prototypes', 'normalized-prototypes']:
        saved = read(f'results/round2/{name}.json')
        assert len(saved['real_small_tables']) == 15
        assert sum(len(row['exact']) for row in saved['real_small_tables']) == 48
        assert saved['summary']['optimal_count'] == 48
        checks[name+'_classwise_milps'] = 48
    for name in ['relative-pilot', 'normalized-pilot']:
        saved = read(f'results/round2/{name}.json')
        assert len(saved['rows']) == 15
        for row in saved['rows']:
            assert not set(row['train_indices']) & set(row['test_indices'])
            for config in row['configurations']:
                assert config['certificate']['certified']
        checks[name+'_disjoint_splits_and_accepted_certificates'] = True
    audit = read('results/round2/normalized-subset-audit.json')
    triage = read('results/round2/normalized-subset-audit-triage.json')
    assert audit['valid_frozen_run'] and triage['valid_frozen_run']
    for run in [audit, triage]:
        assert run['summary']['fully_classified']
        assert len(run['rows']) == 730
    states = lambda run: {(r['dataset'], tuple(r['features'])): r['status'] for r in run['rows']}
    assert states(audit) == states(triage)
    assert audit['summary']['status_counts'] == {'infeasible': 632, 'feasible': 98}
    assert audit['summary']['milp_class_calls'] == audit['summary']['milp_optimal_calls'] == 2528
    checks['all_milp_matches_triage_all_730_statuses'] = True
    paths = read('results/round2/normalized-subset-paths.json')
    assert paths['source_sha256'] == sha(ROOT/paths['source_file'])
    assert paths['datasets']['wdbc']['true_feasible']['minimum_cardinality_including_full'] == 1
    assert paths['datasets']['wdbc']['true_feasible']['minimum_reachable_cardinality'] == 3
    checks['paths_source_and_wdbc_gap'] = True
    candidate = read('results/round2/normalized-candidate-pilot.json')
    assert candidate['run_status'] == 'complete' and candidate['valid_frozen_run']
    assert candidate['summary']['certificates_evaluated'] == 431
    assert len(candidate['rows']) == 15
    assert candidate['summary']['changed_subset_datasets'] == []
    for row in candidate['rows']:
        assert row['selected_certificate']['certified']
        assert row['selected_features'] == row['original_greedy_features']
        selected = next(c for c in row['candidates'] if c['features'] == row['selected_features'])
        eligible = [c for c in row['candidates'] if c['certificate']['certified']]
        rank = lambda c: (len(c['features']), c['certificate']['max_upper_bound'], c['features'])
        assert rank(selected) == min(map(rank, eligible))
    for path, expected in candidate['source_hashes_before'].items():
        assert sha(ROOT/path) == expected
    assert sha(ROOT/'research/round2/normalized-candidate-pilot.md') == candidate['protocol_note_sha256_before']
    assert sum(len(r['evaluations']) for r in candidate['rows']) == 495
    checks['431_direct_candidates_follow_frozen_training_selection_rule'] = True
    tests = read('results/round2/test-run.json')
    assert tests['exit_code'] == 0 and tests['sources_unchanged']
    for path, expected in tests['source_hashes'].items():
        assert sha(ROOT/path) == expected
    checks['test_sources_still_match'] = tests['summary']
    invalid = read('results/round2/archive/normalized-pilot-incomplete-v0-status.json')
    assert invalid['run_status'] == 'invalid'
    assert invalid['sha256'] == sha(ROOT/'results/round2/archive'/invalid['artifact'])
    checks['invalid_attempt_preserved_and_labelled'] = True
    result['status'] = 'passed'
    result['script_sha256'] = sha(Path(__file__))
    target = ROOT/'results/round2/integration-audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
