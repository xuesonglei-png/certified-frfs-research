"""Matched greedy joint zero faces, all15 exposed first training folds."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from run_frs_zero_face_repair_experiment import prerequisites, builtin
from frs_joint_zero_face import eliminate
from frs_coordinate_minimum import MODES
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_pair_transition_cell import build_class_cell, evaluate_class_cell as pair_value
from frs_one_sided_cell import evaluate_class_cell as one_value
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

SOURCE = ROOT/'results/frs-coordinate-diagnostic-v2'
FOLDER = ROOT/'results/frs-joint-zero-face'
FREEZE = ROOT/'research/frs-joint-zero-face-source-freeze.json'


def frozen():
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    return hashes


def statistics(saved):
    p, d = len(saved['original_support']), saved['d']
    row = {'dataset':saved['dataset'], 'available':saved['available'], 'original_support_size':p,
        'd':d, 'fit_seconds':saved['fit_seconds'], 'modes':{}}
    for mode in MODES:
        if not saved['available']:
            row['modes'][mode] = None; continue
        r = saved['eliminations'][mode]; zeros = len(r['exact_zero_indices'])
        status = {s:sum(v['status'] == s for v in r['steps']) for s in sorted({v['status'] for v in r['steps']})}
        cell = (one_value if mode == 'one_sided' else pair_value)(saved['geometry']['cell'], r['target']) if mode in ('one_sided', 'pair_transition') else None
        row['modes'][mode] = {'attempted_faces':len(r['steps']), 'accepted_faces':sum(v['accepted'] for v in r['steps']),
            'status_counts':status, 'exact_joint_deletions':zeros, 'final_exact_support_size':p-zeros,
            'additional_attribute_removal_fraction':zeros/p, 'total_attribute_removal_fraction':1-(p-zeros)/d,
            'class_upper_bounds':None if cell is None else cell['class_upper_bounds'],
            'FRS_sufficient_region_feasible':None if cell is None else cell['feasible']}
    return row


def run(name):
    threadpool_limits(1); audit = prerequisites(); hashes = frozen(); FOLDER.mkdir(exist_ok=True)
    path = FOLDER/(name+'.json')
    if path.exists():
        saved = json.loads(path.read_text()); assert saved['complete'] and saved['fingerprints'] == hashes
        assert saved['source_sha256'] == digest(SOURCE/(name+'.json')); return
    source_path = SOURCE/(name+'.json'); assert audit['artifact_hashes'][name] == digest(source_path)
    source = json.loads(source_path.read_text()); X,y = load_dataset(name)
    ri,ca = np.asarray(source['reference_indices']),np.asarray(source['training_indices'])
    A,_ = preprocess_fit(X[ri][ca],X[ri][ca]); timer = time.perf_counter()
    geometry = None; eliminations = None
    if source['strict_cell_available']:
        geometry = source['diagnostic']['geometry']; fs = geometry['features']; anchor = np.asarray(geometry['anchor'])
        C,rhs,_ = anchor_witness_polytope(A,y[ri][ca],fs,anchor)
        SC,SR,cell = build_class_cell(A,y[ri][ca],fs,anchor,np.asarray(geometry['cell']['full_weights']))
        assert cell == geometry['cell']
        eliminations = {m:eliminate(geometry,C,rhs,SC,SR,m) for m in MODES}
    saved = {'dataset':name, 'd':source['d'], 'fold':0, 'available':source['strict_cell_available'],
        'reference_indices':source['reference_indices'], 'training_indices':source['training_indices'],
        'original_support':source['original_support'], 'original_anchor':source['original_anchor'],
        'source_sha256':digest(source_path), 'fingerprints':hashes, 'geometry':geometry, 'eliminations':eliminations,
        'fit_seconds':time.perf_counter()-timer, 'complete':True, 'classification_scored':False, 'goal_achieved':False}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'matched actual joint supports',{m:None if v is None else v['final_exact_support_size']
        for m,v in statistics(saved)['modes'].items()},flush=True)


def summarize(names):
    rows = [statistics(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    counts = {m:{k:sum(r['modes'][m][k] for r in rows if r['available']) for k in
        ('attempted_faces','accepted_faces','exact_joint_deletions')} for m in MODES}
    means = {m:{k:float(np.mean([r['modes'][m][k] for r in rows if r['available']])) for k in
        ('final_exact_support_size','additional_attribute_removal_fraction','total_attribute_removal_fraction')} for m in MODES}
    out = {'rows':rows,'datasets':len(names),'counts':counts,'means':means,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'classification_scored':False,'goal_achieved':False,
        'limits':['Joint zeros are checked together in one actual feasible vector.',
                  'First folds of exposed data do not establish prediction or generalization.',
                  'Greedy failure or no positive interior slack is not global irreducibility.',
                  'Majority-only and stable-only are uncertified FRS diagnostic ablations.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'counts':counts,'means':means},indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names=sorted(prerequisites()['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)
