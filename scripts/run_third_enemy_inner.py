"""ALL15/44 common DC target development; no outer scoring or label API."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from third_enemy_hinge_stage import GRID, PredictiveMetricStage
from geometric_vote_evaluation import evaluate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

SOURCE = ROOT/'results/atomic-pruning-classifier-v2'
PREVIOUS = ROOT/'results/majority-hinge-inner-all15'
FOLDER = ROOT/'results/third-enemy-inner'
FREEZE = ROOT/'research/third-enemy-inner-source-freeze.json'
METHODS = ('proposed', 'unguarded_refined', 'raw_repair_refined', 'full_refined', 'fgs_refined', 'sfrip_refined')


def frozen():
    freeze = json.loads(FREEZE.read_text())
    for rel, sha in freeze['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    return freeze['fingerprints']


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    hashes = {**frozen(), (SOURCE/(name+'.json')).relative_to(ROOT).as_posix(): digest(SOURCE/(name+'.json')),
              (PREVIOUS/(name+'.json')).relative_to(ROOT).as_posix(): digest(PREVIOUS/(name+'.json'))}
    assert verify(SOURCE)['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    assert verify(PREVIOUS)['artifact_hashes'][name] == digest(PREVIOUS/(name+'.json'))
    out = FOLDER/(name+'.json')
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    source = json.loads((SOURCE/(name+'.json')).read_text())
    X, y = load_dataset(name); ri = np.asarray(source['reference_indices']); raw, labels = X[ri], y[ri]
    inner = []
    for fi, b in enumerate(source['inner']):
        ca, cb = np.asarray(b['reference_indices']), np.asarray(b['validation_indices'])
        assert not set(ca)&set(cb)
        part = FOLDER/(name+f'.inner{fi}.partial.json')
        if part.exists():
            stored = json.loads(part.read_text()); assert stored['fingerprints'] == hashes; inner.append(stored['block']); continue
        A, B = preprocess_fit(raw[ca], raw[cb]); stage = PredictiveMetricStage(A, labels[ca]); start = time.perf_counter()
        bases = {m: {k: b['options'][m][source['selection']['selected'][m]]['metric'][k] for k in ('features', 'weights')}
                 | {'origin': 'frozen_atomic_v2_selected_metric'} for m in METHODS}
        options, records, lookup = {}, [], {}
        for m in METHODS:
            options[m] = []
            for ti in range(len(GRID)):
                metric = stage.metric(bases[m], ti, certified=m == 'proposed'); key = (tuple(metric['features']), tuple(metric['weights']))
                if key not in lookup:
                    lookup[key] = len(records); records.append({'metric': metric, 'evaluation': evaluate(A, labels[ca], B, labels[cb], metric)})
                options[m].append({'target_index': ti, 'metric': metric, 'record_id': lookup[key]})
        block = {'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'bases': bases,
                 'stage': stage.export(), 'options': options, 'records': records, 'fit_score_seconds': time.perf_counter()-start}
        part.write_text(json.dumps({'fingerprints': hashes, 'block': block}, separators=(',', ':')), encoding='utf-8')
        inner.append(block)
        print(name, 'inner', fi+1, 'DC learners', len(stage.learners), 'subproblem failures',
              sum(r['fit']['subproblem_failures'] for r in stage.learners), flush=True)
    selection = {}
    for m in METHODS:
        rows = []
        for ti in range(len(GRID)):
            values = [b['records'][b['options'][m][ti]['record_id']]['evaluation']['metrics'] for b in inner]
            rows.append({'target_index': ti, 'mean_BA': float(np.mean([r['clean_BA'] for r in values])),
                         'mean_logloss': float(np.mean([r['balanced_logloss'] for r in values]))})
        chosen = min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_BA'], 12), round(rows[i]['mean_logloss'], 12), i))
        selection[m] = {'options': rows, 'selected': chosen, 'selected_inner_gain_pp': 100*(rows[chosen]['mean_BA']-rows[0]['mean_BA'])}
    saved = {'dataset': name, 'fingerprints': hashes, 'reference_indices': ri.tolist(), 'inner': inner, 'selection': selection,
             'complete': True, 'goal_achieved': False, 'new_outer_evaluation_performed': False,
             'scope': 'common_DC_loss_all15_exposed_inner_only_sequential_frozen_support_refinement'}
    out.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')
    print(name, 'INNER COMPLETE gain pp', selection['proposed']['selected_inner_gain_pp'], flush=True)


def summarize(names):
    rows, failures = [], 0
    for name in names:
        r = json.loads((FOLDER/(name+'.json')).read_text()); old = json.loads((PREVIOUS/(name+'.json')).read_text())
        rows.append({'dataset': name, 'anchor_gain_pp': {m: r['selection'][m]['selected_inner_gain_pp'] for m in METHODS},
                     'vs_convex_pp': {m: 100*(r['selection'][m]['options'][r['selection'][m]['selected']]['mean_BA']
                                               -old['selection'][m]['options'][old['selection'][m]['selected']]['mean_BA']) for m in METHODS}})
        failures += sum(l['fit']['subproblem_failures'] for b in r['inner'] for l in b['stage']['learners'])
    mean_anchor = {m: float(np.mean([r['anchor_gain_pp'][m] for r in rows])) for m in METHODS}
    mean_previous = {m: float(np.mean([r['vs_convex_pp'][m] for r in rows])) for m in METHODS}
    wins = sum(r['vs_convex_pp']['proposed'] > 1e-10 for r in rows)
    summary = {'rows': rows, 'datasets': len(names), 'inner_folds': sum(len(json.loads((FOLDER/(n+'.json')).read_text())['inner']) for n in names),
        'mean_anchor_gain_pp': mean_anchor, 'mean_vs_convex_pp': mean_previous, 'improved_vs_convex_datasets': wins,
        'target_optimizer_failures': failures, 'pilot_gate_passed': len(names) == 15 and mean_previous['proposed'] >= .5 and wins >= 6 and failures == 0,
        'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names}, 'independent_audit_required': True,
        'goal_achieved': False, 'new_outer_evaluation_performed': False, 'scope': 'development_inner_signal_not_prediction_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(verify(SOURCE)['artifact_hashes']); assert len(names) == 15
        with ProcessPoolExecutor(max_workers=2) as executor:
            for f in as_completed([executor.submit(run, n) for n in names]):
                f.result()
        summarize(names)
