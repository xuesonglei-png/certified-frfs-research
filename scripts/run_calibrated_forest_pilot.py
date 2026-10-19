"""Fixed common calibrated forest, new certified teacher path, count controls."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_spline_capacity_pilot import as_spline3
from certified_frfs.calibrated_interaction_forest import CalibratedForestContext
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_forest import spline_scores,spline_box_margins
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-geometry-forest'; FOLDER = ROOT/'results/calibrated-forest-pilot'
METHODS = ('new','old_support','full','fgs_matched','sfrip_matched','fgs_native','sfrip_native','old_original','full_original')

def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/calibrated-forest-pilot-protocol.md',SOURCE/(name+'.json'),
        ROOT/'src/certified_frfs/calibrated_interaction_forest.py',ROOT/'scripts/run_spline_capacity_pilot.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        prior = json.loads(out.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return
        records = prior['records']
    X,y = load_dataset(name); index = 0
    for i,s in enumerate(source['splits']):
        ri = np.array(s['reference_indices'])
        for j,inner in enumerate(s['inner']):
            index += 1
            if index <= len(records): continue
            a,b = np.array(inner['reference_indices']),np.array(inner['validation_indices']); raw,labels = X[ri[a]],y[ri[a]]
            A,B = preprocess_fit(raw,X[ri[b]]); d = A.shape[1]; start = time.perf_counter()
            context = CalibratedForestContext(raw,labels)
            context.geometry = {tuple(g['features']):g for g in inner['geometry']}
            stage = context.fit_stage(C=1.,pair_penalty=1.); candidate = stage['policies']['0.8']['candidate']; fs = candidate['features']; target = len(fs)
            oldstage = inner['stages'][7]; old = inner['models'][oldstage['policies']['0.8']['model_id']]; full = inner['models'][oldstage['policies']['1.0']['model_id']]
            fits = {'new':as_spline3(stage['policies']['0.8']['fit']),
                'old_support':as_spline3(context.classifier(old['features'],1.,1.)),
                'full':as_spline3(context.classifier(list(range(d)),1.,1.)),
                'old_original':as_spline3(old),'full_original':as_spline3(full)}
            selectors = {}
            for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
                path = native['splits'][i]['inner'][j]['paths'][method][ci]; nf,mf = sorted(path['native']['features']),sorted(path['ranking'][:target])
                fits[prefix+'_native'] = as_spline3(context.classifier(nf,1.,1.)); fits[prefix+'_matched'] = as_spline3(context.classifier(mf,1.,1.))
                selectors[method] = {'selector_index':ci,'config':GRIDS[method][ci],'ranking':path['ranking'],'native_features':nf,'matched_features':mf}
            seconds = time.perf_counter()-start
            exact = all(len(fits[m]['features']) == target for m in ('fgs_matched','sfrip_matched'))
            # All calibration, attribute selections and fits finalized above.
            rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            metrics,boxes = {},{}
            for method,fit in fits.items():
                values = [balanced_accuracy(spline_scores(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
                box = spline_box_margins(B,y[ri[b]],fit,eps=.01,witnesses=True)
                boxes[method] = {'margins':box['margins'].tolist(),'rivals':box['rivals'].tolist(),'witness_raw':box['witness_raw'].tolist()}
                metrics[method] = {'clean':values[0],'noise':float(np.mean(values[1:])),
                    'robust':coverage(box['margins'] > 1e-10,y[ri[b]]),'removed':1-len(fit['features'])/d}
            records.append({'outer_index':i,'inner_index':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
                'validation_indices':b.tolist(),'query_seed':144+i,'target_count':target,'counts_exact':exact,'seconds':seconds,
                'calibration':context.calibration,'stage':stage,'selector_paths':selectors,'fits':fits,'metrics':metrics,'boxes':boxes})
            out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,
                'complete':len(records) == sum(len(s['inner']) for s in source['splits']),
                'scope':'common_calibrated_sparse_forest_original_FRS_contract_exposed_development'},separators=(',',':')),encoding='utf-8')
            print(name,i,j,{m:round(v['clean'],5) for m,v in metrics.items()},'cal_edges',len(context.calibration['edges']),
                'count',target,'old',len(old['features']),flush=True)

def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'optimizer_caps':0,'changed_supports':0,'zero_edge_references':0,'selected_edges':0,'eligible_edges':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']; counts['optimizer_caps'] += sum(not f['optimizer_converged'] for f in r['fits'].values())
            counts['changed_supports'] += r['fits']['new']['features'] != r['fits']['old_original']['features']; counts['zero_edge_references'] += not r['calibration']['edges']
            counts['selected_edges'] += len(r['calibration']['edges']); counts['eligible_edges'] += sum(c['eligible'] for c in r['calibration']['candidates'])
            assert r['stage']['policies']['0.8']['candidate']['certificate']['feasible']
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','robust','removed')} for m in METHODS}})
    assert len(rows) == 15 and counts['records'] == 128
    result = {'rows':rows,'counts':counts,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','removed')} for m in METHODS},
        'contrasts':{metric:{m:paired([r['methods']['new'][metric]-r['methods'][m][metric] for r in rows]) for m in METHODS[1:]} for metric in ('clean','noise','robust')},
        'scope':'all15_all128_common_sparse_classifier_original_FRS_fixed_recent_settings_development'}
    clean = result['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); result['matched_holm'] = corrected
    result['development_numeric_gate'] = bool(all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched'))
        and clean['full']['mean_pp'] >= 0 and result['means']['new']['removed'] >= .1 and counts['count_mismatches'] == counts['optimizer_caps'] == 0)
    result['goal_achieved'] = False; result['independent_audit_required'] = True
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'rows'},flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
