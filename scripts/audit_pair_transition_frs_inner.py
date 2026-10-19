"""Independent new-cell audits and full native-family selection replay."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_pair_transition_frs_inner import (ROOT, SOURCE, PREVIOUS, CANONICAL, FOLDER,
    METHODS, STRONG, digest, frozen, prerequisites)
from audit_pair_transition_inner_block import check_append_block
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); audits = prerequisites(); counts = Counter(datasets=1)
    saved = json.loads((FOLDER/(name+'.json')).read_text()); hashes = frozen()
    assert saved['fingerprints'] == hashes and saved['complete'] and not saved['new_outer_evaluation_performed']
    prior = json.loads((PREVIOUS/(name+'.json')).read_text())
    atomic = json.loads((SOURCE/(name+'.json')).read_text())
    canonical = json.loads((CANONICAL/(name+'.json')).read_text())
    assert saved['canonical_inner_source_sha256'] == audits[CANONICAL.name]['artifact_hashes'][name]
    assert saved['reference_indices'] == prior['reference_indices'] == atomic['reference_indices'] == canonical['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    for fi, (b, p, a, c) in enumerate(zip(saved['inner'], prior['inner'], atomic['inner'], canonical['inner'], strict=True)):
        for key in ('reference_indices', 'validation_indices'):
            assert b[key] == p[key] == a[key] == c[key]
        ca, cb = np.asarray(b['reference_indices']), np.asarray(b['validation_indices'])
        assert not set(ca)&set(cb); A, B = preprocess_fit(raw[ca], raw[cb])
        for m in METHODS:
            anchor = a['options'][m][atomic['selection']['selected'][m]]['metric']
            equivalent({k:anchor[k] for k in ('features', 'weights')}, {k:b['bases'][m][k] for k in ('features', 'weights')})
        check_append_block(A, labels[ca], B, labels[cb], b, p, counts)
        print(name, 'inner', fi+1, 'append-only pair fits/duals/guards/votes replayed', flush=True)
    chosen = {}
    for m in METHODS:
        rows = []
        for ti in range(35):
            v = [b['records'][b['options'][m][ti]['record_id']]['evaluation']['metrics'] for b in saved['inner']]
            rows.append({'target_index':ti, 'mean_BA':float(np.mean([r['clean_BA'] for r in v])),
                'mean_logloss':float(np.mean([r['balanced_logloss'] for r in v]))})
        i = min(range(35), key=lambda j:(-round(rows[j]['mean_BA'], 12), round(rows[j]['mean_logloss'], 12), j))
        equivalent(saved['selection'][m], {'options':rows, 'selected':i,
            'selected_inner_gain_pp':100*(rows[i]['mean_BA']-rows[0]['mean_BA'])})
        chosen[m] = rows[i]['mean_BA']
    for m in STRONG:
        rows = []
        for i, op in enumerate(canonical['inner'][0]['options'][m]):
            rs = [b['records'][b['options'][m][i]['record_id']] for b in canonical['inner']]
            rows.append({'config_index':op['config_index'], 'target_index':op['target_index'],
                'mean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in rs])),
                'mean_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in rs])),
                'mean_count':float(np.mean([len(r['metric']['features']) for r in rs]))})
        i = min(range(len(rows)), key=lambda j:(-round(rows[j]['mean_BA'], 12),
                    round(rows[j]['mean_logloss'], 12), rows[j]['mean_count'], j))
        equivalent(canonical['selection'][m], {'options':rows, 'selected_index':i})
        equivalent(saved['strong_controls'][m], {'selected_index':i, 'candidate_count':len(rows),
            'selected_recipe':rows[i], 'provenance':'entire_audited_canonical_inner_candidate_family'})
        chosen[m] = rows[i]['mean_BA']; counts['audited_native_candidate_options_reused'] += len(rows)
    row = {'dataset':name,
        'anchor_gain_pp':{m:saved['selection'][m]['selected_inner_gain_pp'] for m in METHODS},
        'vs_previous_DC_pp':{m:100*(chosen[m]-prior['selection'][m]['options'][prior['selection'][m]['selected']]['mean_BA']) for m in METHODS},
        'selected_inner_BA':chosen, 'selected_new_pair_family':{m:saved['selection'][m]['selected'] >= 28 for m in METHODS}}
    return {'dataset':name, 'counts':dict(counts), 'row':row, 'sha256':digest(FOLDER/(name+'.json'))}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        r = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True, **r}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    reports.sort(key=lambda r:r['dataset']); rows = [r['row'] for r in reports]
    equivalent(rows, summary['rows']); assert counts['datasets'] == 15 and counts['inner_folds'] == summary['inner_folds'] == 44
    assert counts['target_optimizer_failures'] == summary['target_optimizer_failures']
    gain = {m:float(np.mean([r['vs_previous_DC_pp'][m] for r in rows])) for m in METHODS}
    anchor = {m:float(np.mean([r['anchor_gain_pp'][m] for r in rows])) for m in METHODS}
    delta = {m:float(np.mean([100*(r['selected_inner_BA']['proposed']-r['selected_inner_BA'][m]) for r in rows])) for m in METHODS[1:]+STRONG}
    wins = sum(r['vs_previous_DC_pp']['proposed'] > 1e-10 for r in rows)
    equivalent(gain, summary['mean_vs_previous_DC_pp']); equivalent(anchor, summary['mean_anchor_gain_pp'])
    equivalent(delta, summary['mean_proposed_minus_controls_pp']); assert wins == summary['improved_vs_previous_DC_datasets']
    assert summary['pilot_gate_passed'] == bool(gain['proposed'] >= .5 and wins >= 6 and counts['target_optimizer_failures'] == 0
        and all(delta[m] > 0 for m in ('full_refined', 'fgs_refined', 'sfrip_refined')+STRONG))
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    out = {'passed':True, 'counts':dict(counts), 'artifact_hashes':hashes,
        'summary_sha256':digest(FOLDER/'complete.json'), 'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False, 'new_outer_evaluation_performed':False,
        'scope':'independent_append_only_geometry_LP_guards_votes_and_entire_native_selection_not_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
