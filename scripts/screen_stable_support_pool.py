"""Fixed diverse certified pool and third-partition resource gate, reference only."""
import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.crossfit_rbf_refit import CrossfitRBFRefit
from certified_frfs.stable_support_pool import support_pool,select_stable_pool

INPUT = ROOT/'results/crossfit-rbf-refit-screen'; FOLDER = ROOT/'results/stable-support-pool-screen'


def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source); X,y = load_dataset(name)
    upstream = json.loads((INPUT/(name+'.json')).read_text()); audit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert audit['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    assert json.loads((INPUT/'complete-audit.json').read_text())['passed']
    hashes = dict(upstream['fingerprints'])
    for p in (Path(__file__),ROOT/'research/stable-support-pool-screen-protocol.md',ROOT/'src/certified_frfs/stable_support_pool.py',
        INPUT/(name+'.json'),INPUT/(name+'-audit.json')): hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    for rel,sha in hashes.items(): assert digest(ROOT/rel) == sha
    artifact = FOLDER/(name+'.json'); rows = []
    if artifact.exists():
        prior = json.loads(artifact.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return
        rows = prior['rows']
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    for index,((i,j,s,inner),old) in enumerate(zip(expected,upstream['rows'])):
        if index < len(rows): continue
        ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; start = time.perf_counter()
        assert old['outer'] == i and old['inner'] == j and old['reference_indices'] == a.tolist()
        initial = old['path']['initial_features']; target = len(initial); proposals = []; provenance = []
        for method,grid in GRIDS.items():
            for ci,config in enumerate(grid):
                ref = native['splits'][i]['inner'][j]['paths'][method][ci]; features = sorted(ref['ranking'][:target])
                origin = method+'_'+str(ci); proposals.append((origin,features))
                provenance.append({'origin':origin,'method':method,'index':ci,'config':config,'ranking':ref['ranking'],'features':features})
        pool = support_pool(initial,raw.shape[1],proposals)
        context = GeometryStageContext(raw,labels); context.geometry = {tuple(g['features']):g for g in inner['geometry']}
        for attempt in old['path']['attempts']:
            geometry = attempt['candidate']; context.geometry[tuple(geometry['features'])] = geometry
        first = CrossfitRBFRefit(raw,labels,seed=20261020); second = CrossfitRBFRefit(raw,labels,seed=20261025)
        inherited = {tuple(e['features']):e['risk'] for e in old['risk_evaluations']}; first.cache = inherited.copy()
        path = select_stable_pool(initial,pool,[first,second],context.candidate)
        third = CrossfitRBFRefit(raw,labels,seed=20261026)
        checks = {'old':third(initial),'new':third(path['features'])}
        minimum_gain = min(b['balanced_error']-v['balanced_error'] for b,v in zip(path['initial_risk_by_partition'],path['final_risk_by_partition']))
        check_gain = checks['old']['balanced_error']-checks['new']['balanced_error']
        folds = {str(r.seed):[{k:f[k] for k in ('reference_indices','validation_indices')} for f in r.folds] for r in (first,second,third)}
        counts = {'pool_size':len(pool),'first_reused':sum(tuple(r['features']) in inherited for r in pool),
            'first_new_refits':sum(len(first.folds) for r in pool if tuple(r['features']) not in inherited),
            'second_refits':len(pool)*len(second.folds),'third_refits':len(third.cache)*len(third.folds)}
        rows.append({'outer':i,'inner':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),'path':path,
            'provenance':provenance,'folds':folds,'third_check':checks,'minimum_selection_gain_pp':100*minimum_gain,
            'third_check_gain_pp':100*check_gain,'counts':counts,'seconds':time.perf_counter()-start})
        artifact.write_text(json.dumps({'dataset':name,'fingerprints':hashes,'rows':rows,'complete':len(rows) == len(expected),
            'archived_query_labels_used':False,'scope':'adaptive_two_partition_pool_third_same_data_check'},separators=(',',':')),encoding='utf-8')
        print(name,i,j,'minimum selection pp',round(100*minimum_gain,6),'third check pp',round(100*check_gain,6),
            'changed',path['features'] != initial,'pool',len(pool),'seconds',round(rows[-1]['seconds'],2),flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((INPUT/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]): f.result()
    counts = Counter(); rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete'] and not saved['archived_query_labels_used']
        for r in saved['rows']:
            path = r['path']; counts['references'] += 1; counts['changed_supports'] += path['features'] != path['initial_features']
            counts['third_improved_references'] += r['third_check_gain_pp'] > 1e-10
            counts['count_mismatches'] += len(path['features']) != len(path['initial_features'])
            counts['attempts'] += len(path['attempts']); counts['rejections'] += sum(not a['accepted'] for a in path['attempts'])
            counts.update(r['counts'])
            counts['status_failures'] += sum(f['fit_status'] != 0 for entry in path['scored_pool'] for v in entry['risk_by_partition'] for f in v['fold_fits'])
            counts['status_failures'] += sum(f['fit_status'] != 0 for v in r['third_check'].values() for f in v['fold_fits'])
            assert path['candidate']['certificate']['feasible']
        rows.append({'dataset':name,'minimum_selection_gain_pp':float(np.mean([r['minimum_selection_gain_pp'] for r in saved['rows']])),
            'third_check_gain_pp':float(np.mean([r['third_check_gain_pp'] for r in saved['rows']]))})
    assert len(rows) == 15 and counts['references'] == 128
    minimum,check = (float(np.mean([r[k] for r in rows])) for k in ('minimum_selection_gain_pp','third_check_gain_pp'))
    gate = minimum >= .5 and check >= .5 and counts['third_improved_references'] >= 20 and counts['count_mismatches'] == counts['status_failures'] == 0
    result = {'rows':rows,'counts':dict(counts),'mean_minimum_selection_gain_pp':minimum,'mean_third_check_gain_pp':check,
        'numeric_resource_gate':bool(gate),'independent_audit_required':True,'goal_achieved':False,'archived_query_labels_used':False,
        'scope':'training_pool_stability_not_prediction_superiority_or_independent_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
