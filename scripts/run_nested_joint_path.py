"""Fresh inner scores choose certified original/joint paths and depth."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_geometry_forest import ROOT, MODELS, FRACTIONS, Registry, digest, holm
from certified_frfs.datasets import load_dataset
from certified_frfs.joint_deletion_geometry import JointDeletionGeometryContext
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/nested-joint-path'
METHODS = ('proposed','original','full','fgs2023','sfrip2025')


def source_verified(name):
    path = SOURCE/(name+'.json'); saved = json.loads(path.read_text()); assert saved['complete']
    audit = json.loads((SOURCE/'complete-audit.json').read_text()); assert audit['passed']
    per = json.loads((SOURCE/(name+'-audit.json')).read_text())
    assert per['artifact_sha256'] == digest(path)
    assert per['auditor_sha256'] == digest(ROOT/'scripts/audit_nested_geometry_forest.py')
    for rel,value in saved['fingerprints'].items(): assert digest(ROOT/rel) == value
    return saved


def seed_context(raw, y, block):
    context = JointDeletionGeometryContext(raw,y)
    context.geometry = {tuple(g['features']):g for g in block['geometry']}
    context.trees = {tuple(t['features']):t for t in block['trees']}
    for fit in block['models']:
        fs = tuple(fit['features']); gamma = fit['pair_penalty'] if fit['edges'] else None
        context.classifiers[(fs,float(fit['C']),gamma)] = fit
    # One-attribute supports have no pair edge regardless of pair option.
    stages = block['stages']; stages = list(stages.values()) if isinstance(stages,dict) else stages
    for stage in stages:
        for policy in stage['policies'].values():
            fit = block['models'][policy['model_id']]
            context.classifiers[(tuple(fit['features']),stage['C'],stage['pair_penalty'])] = fit
    return context


def packed_joint(registry, stage):
    packed = registry.stage(stage)
    for key in ('pair_pool_size','accepted_pair_jumps'): packed['path'][key] = stage['path'][key]
    for actual, expected in zip(packed['path']['rankings'],stage['path']['rankings']):
        assert actual == expected
    return packed


def choices():
    return [{'family':family,'family_index':family_index,'model_index':mi,'fraction':f,'fraction_index':fi,'folds':[]}
        for family_index,family in enumerate(('original','joint')) for mi in range(len(MODELS)) for fi,f in enumerate(FRACTIONS)]


def choose(rows):
    for row in rows:
        row['mean_clean'] = float(np.mean([v['clean'] for v in row['folds']]))
        row['mean_size'] = float(np.mean([len(v['features']) for v in row['folds']]))
    return max(range(len(rows)),key=lambda j:(round(rows[j]['mean_clean'],12),-rows[j]['mean_size'],
        -rows[j]['model_index'],-rows[j]['fraction_index'],-rows[j]['family_index']))


def run(name):
    threadpool_limits(1); source = source_verified(name)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/nested-joint-path-protocol.md',SOURCE/(name+'.json'),
            ROOT/'src/certified_frfs/joint_deletion_geometry.py',ROOT/'scripts/audit_nested_geometry_forest.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']: return
        records = saved['splits']
    X,y = load_dataset(name)
    for i,s in enumerate(source['splits']):
        if i < len(records): continue
        ri,vi = np.array(s['reference_indices']),np.array(s['validation_indices'])
        rows = choices(); inners = []
        for j,old in enumerate(s['inner']):
            a,b = np.array(old['reference_indices']),np.array(old['validation_indices'])
            assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
            context = seed_context(X[ri[a]],y[ri[a]],old); _,B = preprocess_fit(X[ri[a]],X[ri[b]])
            registry = Registry(context,B,y[ri[b]]); stages = []
            for mi,cfg in enumerate(MODELS):
                joint = context.fit_stage(**cfg,pair_pool_size=8); packed = packed_joint(registry,joint); stages.append(packed)
                for fi,f in enumerate(FRACTIONS):
                    new = joint['policies'][str(f)]; mid,clean = registry.score(new['fit'])
                    rows[48+4*mi+fi]['folds'].append({'model_id':mid,'features':new['candidate']['features'],'clean':clean})
                    policy = old['stages'][mi]['policies'][str(f)]
                    fit = old['models'][policy['model_id']]; mid,clean = registry.score(fit)
                    rows[4*mi+fi]['folds'].append({'model_id':mid,'features':fit['features'],'clean':clean})
            inners.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'stages':stages,**registry.export()})
            print(name,i,'joint inner',j,'models',len(registry.models),'geometry',len(context.geometry),flush=True)
        selected = choose(rows); chosen = rows[selected]; cfg = MODELS[chosen['model_index']]
        context = seed_context(X[ri],y[ri],s); _,B = preprocess_fit(X[ri],X[vi]); registry = Registry(context)
        if chosen['family'] == 'joint': stage = context.fit_stage(**cfg,pair_pool_size=8)
        else:
            from certified_frfs.geometry_only_forest_stage import GeometryStageContext
            stage = GeometryStageContext.fit_stage(context,**cfg)
        packed = packed_joint(registry,stage) if chosen['family'] == 'joint' else registry.stage(stage)
        policy = stage['policies'][str(chosen['fraction'])]
        results = {'proposed':{'chosen_index':selected,'model_index':chosen['model_index'],'fraction':chosen['fraction'],
            'family':chosen['family'],'features':policy['candidate']['features'],'model_id':registry.capture(policy['fit']),
            'geometry_id':packed['policies'][str(chosen['fraction'])]['geometry_id']}}
        for m,source_method in (('original','proposed'),('full','full'),('fgs2023','fgs2023'),('sfrip2025','sfrip2025')):
            old_result = s['results'][source_method]; fit = s['models'][old_result['model_id']]
            results[m] = {'source_method':source_method,'source_model_id':old_result['model_id'],'features':fit['features'],'model_id':registry.capture(fit)}
        # All selected supports and models are set before evaluating labels.
        rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,r in results.items():
            fit = registry.models[r['model_id']]
            values = [balanced_accuracy(orthogonal_scores(Q,fit),fit['classes'],y[vi]) for Q in queries]
            r['metrics'] = {'clean':values[0],'noise':float(np.mean(values[1:]))}
            if m != 'proposed':
                np.testing.assert_allclose([r['metrics'][v] for v in ('clean','noise')],
                    [s['results'][r['source_method']]['metrics'][v] for v in ('clean','noise')],atol=1e-12)
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner':inners,'trace':rows,
            'results':results,'stages':{'proposed':packed},'query_seed':144+i,**registry.export()})
        out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'fingerprints':hashes,'splits':records,
            'complete':len(records) == len(source['splits']), 'scope':'fresh_inner_selected_original_or_joint_certified_path_exposed_development'},separators=(',',':')),encoding='utf-8')
        print(name,i,'joint outer',chosen['family'],chosen['model_index'],chosen['fraction'],{m:r['metrics']['clean'] for m,r in results.items()},flush=True)


def summarize(names):
    rows = []; counts = {'outer_splits':0,'inner_splits':0,'models':0,'optimizer_caps':0,'selected_joint':0,'selected_full':0,'inner_pair_jumps':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        methods = {m:{metric:float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']])) for metric in ('clean','noise')} for m in METHODS}
        for m in METHODS: methods[m]['removed'] = float(np.mean([1-len(s['results'][m]['features'])/saved['d'] for s in saved['splits']]))
        for s in saved['splits']:
            counts['outer_splits'] += 1; counts['inner_splits'] += len(s['inner'])
            counts['selected_joint'] += s['results']['proposed']['family'] == 'joint'
            counts['selected_full'] += len(s['results']['proposed']['features']) == saved['d']
            for b in s['inner']+[s]:
                counts['models'] += len(b['models']); counts['optimizer_caps'] += sum(not m['optimizer_converged'] for m in b['models'])
            counts['inner_pair_jumps'] += sum(st['path']['accepted_pair_jumps'] for b in s['inner'] for st in b['stages'])
        rows.append({'dataset':name,'methods':methods})
    assert len(rows) == 15 and counts['outer_splits'] == 44 and counts['inner_splits'] == 128
    result = {'rows':rows,'counts':counts,'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','removed')} for m in METHODS},
        'contrasts':{metric:{m:paired([r['methods']['proposed'][metric]-r['methods'][m][metric] for r in rows]) for m in METHODS[1:]} for metric in ('clean','noise')},
        'scope':'all15_all44_all128_training_selected_original_joint_paths_exposed_development_not_final_SOTA_confirmation'}
    result['descriptive_holm_p'] = holm({m:result['contrasts']['clean'][m]['sign_flip_two_sided_p'] for m in ('full','fgs2023','sfrip2025')})
    result['gate_passed'] = all(result['contrasts']['clean'][m]['mean_pp'] >= .5 for m in ('fgs2023','sfrip2025')) and result['contrasts']['clean']['full']['mean_pp'] >= 0 and result['means']['proposed']['removed'] >= .1 and counts['optimizer_caps'] == 0
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'rows'},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
