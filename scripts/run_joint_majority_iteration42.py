"""Anchored certified repair on immutable41 training fits, never outer tuning."""
import json, hashlib, time, argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.repaired_majority import repair_majority_metric
from certified_frfs.validated_pool import preprocess_fit, validation_scores


DATASETS = ['banknote', 'ecoli', 'glass', 'haberman', 'heart_statlog', 'ionosphere',
    'iris', 'seeds', 'sonar', 'transfusion', 'vehicle', 'vertebral', 'wdbc', 'wine', 'yeast']


def repair_candidates(A, y, raw):
    mass = np.sqrt(A.shape[1]); rows = []
    for record in raw:
        # Explicit whitelist excludes old validation/test scores from fitting.
        c = {k: record[k] for k in ['config', 'features', 'weights', 'full_weights', 'certificate', 'fit']}
        ridge = c['config']['ridge']
        if ridge == 'original': rows.append(c); continue
        dense = next(r for r in raw if r['config']['ridge'] == ridge and r['config']['fraction'] == 1.)
        rows.append(repair_majority_metric(A, y, c, np.asarray(dense['weights'])/mass, ridge=ridge, anchor_reg=1.))
    assert all(r['certificate']['feasible'] for r in rows)
    return rows


def run(job):
    threadpool_limits(1); name, fold = job
    folder = ROOT/'results/predictive-iteration42'; folder.mkdir(exist_ok=True)
    out = folder/f'{name}-s101-f{fold}.json'
    if out.exists(): raise FileExistsError(out)
    source = ROOT/'results/predictive-iteration41'/out.name
    # Producer publishes JSON with a single write; existence alone can race
    # its final bytes. Admit only a fully parsed immutable byte snapshot.
    for attempt in range(50):
        source_bytes = source.read_bytes()
        try:
            old = json.loads(source_bytes); break
        except json.JSONDecodeError:
            if attempt == 49: raise
            time.sleep(.2)
    started = time.perf_counter(); X, y = load_dataset(name); tr, te = np.array(old['train_indices']), np.array(old['test_indices'])
    data_hash = hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()
    assert old['data_sha256'] == data_hash
    inner = []; trace = {}
    for index, split in enumerate(old['inner_splits']):
        ii, jj = np.array(split['train']), np.array(split['validation'])
        A, B = preprocess_fit(X[tr][ii], X[tr][jj]); rows = repair_candidates(A, y[tr][ii], split['fits'])
        for r in rows:
            scores = validation_scores(A, y[tr][ii], B, y[tr][jj], r['features'], np.asarray(r['weights']), 144+index, joint=True)
            r['validation'] = scores; r['score'] = (scores['clean']+scores['noise01'])/2
            trace.setdefault(json.dumps(r['config'], sort_keys=True), []).append(r)
        inner.append({'train': ii.tolist(), 'validation': jj.tolist(), 'fits': rows})
    configs = [{'config': json.loads(key), 'score': float(np.mean([r['score'] for r in rows])),
        'mean_size': float(np.mean([len(r['features']) for r in rows]))} for key, rows in trace.items()]
    chosen = min(configs, key=lambda c: (-c['score'], c['mean_size'], json.dumps(c['config'], sort_keys=True)))
    F, Q = preprocess_fit(X[tr], X[te]); final = repair_candidates(F, y[tr], old['final_candidate_fits'])
    fitted = next(r for r in final if r['config'] == chosen['config']); assert fitted['certificate']['feasible']
    metrics = validation_scores(F, y[tr], Q, y[te], fitted['features'], np.asarray(fitted['weights']), 104, joint=True)
    rows = [{'method': 'certified_anchored', 'chosen': chosen, 'fit': fitted, 'metrics': metrics}]
    # Controls enter only after the certified configuration/metric is fixed.
    for method in ['sparse', 'unlimited_joint_metric']:
        r = next(r for r in old['rows'] if r['method'] == method)
        scores = validation_scores(F, y[tr], Q, y[te], r['fit']['features'], np.asarray(r['fit']['weights']), 104, joint=True)
        assert all(abs(scores[k]-r['metrics'][k]) < 1e-12 for k in scores)
        rows.append({**r, 'metrics': scores})
    uniform = validation_scores(F, y[tr], Q, y[te], fitted['features'], np.ones(F.shape[1])/np.sqrt(F.shape[1]), 104, joint=True)
    sources = sorted((ROOT/'src/certified_frfs').glob('*.py'))+[ROOT/'scripts/run_joint_majority_iteration42.py', ROOT/'research/joint-majority-iteration42-protocol.md']
    payload = {'dataset': name, 'fold': fold, 'd': X.shape[1], 'train_indices': tr.tolist(), 'test_indices': te.tolist(),
        'inner_splits': inner, 'configurations': configs, 'rows': rows, 'final_candidate_fits': final,
        'uniform_on_selected_support': {'features': fitted['features'], 'metrics': uniform},
        'baseline_rows': old['baseline_rows'], 'seconds_repair_and_scoring_only': time.perf_counter()-started,
        'source41_sha256': hashlib.sha256(source_bytes).hexdigest(), 'data_sha256': data_hash,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scope': 'development_cached_reference_training_fits_certified_anchor_repair_not_untouched_confirmation',
        'timing_limit': 'excludes shared original metric fitting;not total method training time',
        'certificate_protocol': 'original uniform full L2 metric;.1loss/.9retention/.01radius;full mass preserved;anchor penalty1'}
    temporary = out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8'); temporary.replace(out)
    print(name, fold, 'certified_anchored', metrics['clean'], 'size', len(fitted['features']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=2); args = parser.parse_args()
    pilot = json.loads((ROOT/'results/anchored-majority-pilot/verified-analysis.json').read_text()); assert pilot['gate_passed']
    pending = {(n, f) for n in DATASETS for f in range(3)}; active = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            available = sorted(job for job in pending if (ROOT/'results/predictive-iteration41'/f'{job[0]}-s101-f{job[1]}.json').exists())
            for job in available[:max(0, args.workers-len(active))]:
                pending.remove(job); active[pool.submit(run, job)] = job
            if active:
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in done: future.result(); del active[future]
            elif pending:
                # Source generation is a separately verified live job; cached
                # files are admitted only after complete JSON parsing.
                time.sleep(5)
