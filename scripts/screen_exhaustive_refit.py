"""All references, exhaustive neighborhood, no query-label argument."""
import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.crossfit_rbf_refit import CrossfitRBFRefit
from certified_frfs.certified_local_refit import certified_local_refit

SOURCE = ROOT/'results/nested-geometry-forest'; INPUT = ROOT/'results/crossfit-rbf-refit-screen'
FOLDER = ROOT/'results/exhaustive-refit-screen'


def run(name):
    threadpool_limits(1); source = source_verified(name); X,y = load_dataset(name)
    upstream = json.loads((INPUT/(name+'.json')).read_text()); prioraudit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert prioraudit['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    assert json.loads((INPUT/'complete-audit.json').read_text())['passed']
    for rel,sha in upstream['fingerprints'].items(): assert digest(ROOT/rel) == sha
    hashes = dict(upstream['fingerprints'])
    for p in (Path(__file__),INPUT/(name+'.json'),INPUT/(name+'-audit.json'),ROOT/'src/certified_frfs/certified_local_refit.py',
        ROOT/'research/exhaustive-refit-screen-protocol.md',ROOT/'research/certified-local-stationarity-lemma.md'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    artifact = FOLDER/(name+'.json'); rows = []
    if artifact.exists():
        prior = json.loads(artifact.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return
        rows = prior['rows']
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    for index,((i,j,s,inner),old) in enumerate(zip(expected,upstream['rows'])):
        if index < len(rows): continue
        assert old['outer'] == i and old['inner'] == j
        ri = np.array(s['reference_indices']); a = np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; start = time.perf_counter()
        context = GeometryStageContext(raw,labels); context.geometry = {tuple(g['features']):g for g in inner['geometry']}
        for attempt in old['path']['attempts']:
            geometry = attempt['candidate']; context.geometry[tuple(geometry['features'])] = geometry
        initial = old['path']['initial_features']; risk = CrossfitRBFRefit(raw,labels)
        risk.cache = {tuple(e['features']):e['risk'] for e in old['risk_evaluations']}; reused = set(risk.cache)
        path = certified_local_refit(initial,risk,context.candidate)
        evaluations = [{'features':list(fs),'risk':value,'reused_from22':fs in reused} for fs,value in sorted(risk.cache.items())]
        additional = old['path']['final_risk']['balanced_error']-path['final_risk']['balanced_error']
        rows.append({'outer':i,'inner':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
            'folds':old['folds'],'path':path,'experiment22_features':old['path']['features'],
            'additional_internal_gain_pp':100*additional,'risk_evaluations':evaluations,'seconds':time.perf_counter()-start})
        artifact.write_text(json.dumps({'dataset':name,'fingerprints':hashes,'rows':rows,'complete':len(rows) == len(expected),
            'query_labels_used':False,'scope':'reference_only_complete_certified_local_neighborhood'},separators=(',',':')),encoding='utf-8')
        print(name,i,j,'additional internal pp',round(100*additional,6),'steps',path['accepted_steps'],
            'attempts',len(path['attempts']),'new_risks',len(risk.cache)-len(reused),'seconds',round(rows[-1]['seconds'],2),flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]): f.result()
    counts = Counter(); rows = []; seconds = 0.
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete'] and not saved['query_labels_used']
        for r in saved['rows']:
            path = r['path']; counts['references'] += 1; counts['additional_improved_references'] += r['additional_internal_gain_pp'] > 1e-10
            counts['changed_from22'] += path['features'] != r['experiment22_features']; counts['count_mismatches'] += len(path['features']) != len(path['initial_features'])
            counts['attempts'] += len(path['attempts']); counts['rejections'] += sum(not a['accepted'] for a in path['attempts'])
            counts['new_loss_evaluations'] += sum(not e['reused_from22'] for e in r['risk_evaluations'])
            counts['new_refits'] += sum(len(e['risk']['fold_fits']) for e in r['risk_evaluations'] if not e['reused_from22'])
            counts['status_failures'] += sum(f['fit_status'] != 0 for e in r['risk_evaluations'] for f in e['risk']['fold_fits'])
            counts['terminal_references'] += path['termination'] == 'exhaustive_one_exchange_foldwise_stationary'
            assert path['candidate']['certificate']['feasible']; seconds += r['seconds']
        rows.append({'dataset':name,'additional_internal_gain_pp':float(np.mean([r['additional_internal_gain_pp'] for r in saved['rows']]))})
    assert len(rows) == 15 and counts['references'] == 128
    mean = float(np.mean([r['additional_internal_gain_pp'] for r in rows]))
    gate = mean >= .5 and counts['additional_improved_references'] >= 20 and counts['count_mismatches'] == counts['status_failures'] == 0
    result = {'counts':dict(counts),'rows':rows,'mean_additional_internal_gain_pp':mean,'numeric_resource_gate':bool(gate),
        'sum_reference_wall_seconds':seconds,'query_labels_used':False,'goal_achieved':False,'independent_audit_required':True,
        'scope':'additional_opportunity_over22_not_prediction_superiority_or_global_optimality'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
