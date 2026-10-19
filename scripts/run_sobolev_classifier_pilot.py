"""Common train-selected smoothness correction with fixed-count support controls."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import holm
from run_spline_capacity_pilot import as_spline3
from certified_frfs.empirical_product_reduction import marginal_product_anova,feasible_count_refinement
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.recent_budget import fgs_budget_path,sfrip_budget_path
from certified_frfs.sobolev_spline_forest import fit_sobolev_forest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.spline_forest import spline_scores,spline_box_margins
from certified_frfs.forest_penalty_selection import balanced_accuracy
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

INPUT = ROOT/'results/empirical-product-fixed-count'
FOLDER = ROOT/'results/sobolev-classifier-pilot'
METHODS = ('refined','old_selector','full','fgs_matched','sfrip_matched')
ALL_METHODS = METHODS+('refined_alpha0','old_alpha0')
CONFIGS = [{'alpha':alpha,'C':C,'pair':pair,'alpha_index':ai,'C_index':ci,'pair_index':pi}
    for ai,alpha in enumerate((0.,.01,.1)) for ci,C in enumerate((.01,.1,1.)) for pi,pair in enumerate((None,1.))]

def classifier(A,y,fs,edges,config):
    return fit_sobolev_forest(A,y,fs,[] if config['pair'] is None else edges,
        C=config['C'],alpha=config['alpha'],pair_penalty=1.)

def choose(trace,alpha0=False):
    candidates = [i for i,c in enumerate(CONFIGS) if not alpha0 or c['alpha'] == 0]
    return max(candidates,key=lambda i:(round(float(np.mean(trace[i]['scores'])),12),
        -CONFIGS[i]['alpha_index'],-CONFIGS[i]['C_index'],-CONFIGS[i]['pair_index']))

def run(name):
    threadpool_limits(1); source = source_verified(name)
    saved = json.loads((INPUT/(name+'.json')).read_text()); audit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert audit['fingerprints'][name+'.json'] == digest(INPUT/(name+'.json')) and saved['complete']
    assert json.loads((INPUT/'complete-audit.json').read_text())['passed']
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/sobolev-classifier-pilot-protocol.md',INPUT/(name+'.json'),INPUT/(name+'-audit.json'),
        ROOT/'src/certified_frfs/precise_balanced_logistic.py',ROOT/'src/certified_frfs/sobolev_spline_forest.py',
        ROOT/'src/certified_frfs/empirical_product_reduction.py',ROOT/'src/certified_frfs/forest_anova_reduction.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return
        records = previous['records']
    X,y = load_dataset(name)
    for i,s in enumerate(source['splits']):
        if i < len(records): continue
        ri = np.array(s['reference_indices']); qi = np.array(s['validation_indices']); raw,labels = X[ri],y[ri]
        assert set(ri).isdisjoint(qi)
        trace = {m:[{'config_index':ci,'scores':[],'model_ids':[]} for ci in range(len(CONFIGS))] for m in METHODS}; folds = []
        children = [r for r in saved['records'] if r['outer_index'] == i]
        assert len(children) == len(s['inner'])
        for j,(child,inner) in enumerate(zip(children,s['inner'])):
            assert child['outer_reference_indices'] == ri.tolist() and child['reference_indices'] == inner['reference_indices'] and child['validation_indices'] == inner['validation_indices']
            a,b = np.array(child['reference_indices']),np.array(child['validation_indices']); A,B = preprocess_fit(raw[a],raw[b])
            mapping = {'refined':'empirical_refit','old_selector':'old','full':'full','fgs_matched':'fgs_matched','sfrip_matched':'sfrip_matched'}
            models,cache = [],{}
            for method in METHODS:
                prior = child['fits'][mapping[method]]; fs,edges = prior['features'],prior['edges']
                for ci,config in enumerate(CONFIGS):
                    key = (tuple(fs),ci)
                    if key not in cache:
                        fit = classifier(A,labels[a],fs,edges,config); score = balanced_accuracy(spline_scores(B,fit),fit['classes'],labels[b])
                        cache[key] = len(models); models.append({'fit':fit,'clean':score})
                    mid = cache[key]; trace[method][ci]['scores'].append(models[mid]['clean']); trace[method][ci]['model_ids'].append(mid)
            folds.append({'inner_index':j,'reference_indices':a.tolist(),'validation_indices':b.tolist(),'models':models})
        choices = {m:choose(trace[m]) for m in METHODS}
        choices['refined_alpha0'] = choose(trace['refined'],True); choices['old_alpha0'] = choose(trace['old_selector'],True)
        context = GeometryStageContext(raw,labels); stage = context.fit_stage(C=1.,pair_penalty=1.)
        candidate = stage['policies']['0.8']['candidate']; oldfs = candidate['features']; target = len(oldfs); d = context.d
        teacher = as_spline3(context.classifier(list(range(d)),1.,1.)); mass = np.array([1/(len(teacher['classes'])*sum(labels == c)) for c in labels])
        centered,energy,law = marginal_product_anova(teacher,context.A,mass)
        path = feasible_count_refinement(energy,oldfs,context.candidate)
        fgs,sfrip = fgs_budget_path(context.A,labels,beta=.8,xi=1.),sfrip_budget_path(context.A,labels,beta=3.,confidence=.8)
        supports = {'refined':path['features'],'old_selector':oldfs,'full':list(range(d)),
            'fgs_matched':sorted(fgs['ranking'][:target]),'sfrip_matched':sorted(sfrip['ranking'][:target]),
            'refined_alpha0':path['features'],'old_alpha0':oldfs}
        A,B = preprocess_fit(raw,X[qi]); fits = {}
        for method in ALL_METHODS:
            fs = supports[method]; edges = context.tree(fs)['edges'] if fs else []
            fits[method] = classifier(A,labels,fs,edges,CONFIGS[choices[method]])
        # No query label has entered training/selection above.
        rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        metrics,boxes = {},{}
        for method,fit in fits.items():
            values = [balanced_accuracy(spline_scores(Q,fit),fit['classes'],y[qi]) for Q in queries]
            box = spline_box_margins(B,y[qi],fit,eps=.01,witnesses=True)
            boxes[method] = {'margins':box['margins'].tolist(),'rivals':box['rivals'].tolist(),'witness_raw':box['witness_raw'].tolist()}
            metrics[method] = {'clean':values[0],'noise':float(np.mean(values[1:])),
                'robust':coverage(box['margins'] > 1e-10,y[qi]),'removed':1-len(fit['features'])/d}
        exact = all(len(supports[m]) == target for m in ('refined','old_selector','fgs_matched','sfrip_matched'))
        records.append({'outer_index':i,'reference_indices':ri.tolist(),'validation_indices':qi.tolist(),'query_seed':144+i,
            'target_count':target,'counts_exact':exact,'trace':trace,'folds':folds,'choices':choices,'supports':supports,
            'stage':stage,'teacher':teacher,'centered_teacher':centered,'energy':energy,'marginal_law':law,'path':path,
            'fgs_path':fgs,'sfrip_path':sfrip,'fits':fits,'metrics':metrics,'boxes':boxes})
        out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,
            'complete':len(records) == len(source['splits']),'scope':'common_train_tuned_smoothness_and_fixed_count_certified_supports_exposed_development'},separators=(',',':')),encoding='utf-8')
        print(name,i,{m:round(v['clean'],5) for m,v in metrics.items()},'choices',choices,flush=True)

def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'parent_caps':0,'child_caps':0,'child_models':0,'refined_changed':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']; counts['refined_changed'] += r['supports']['refined'] != r['supports']['old_selector']
            counts['parent_caps'] += sum(not f['optimizer_converged'] for f in r['fits'].values())
            for fold in r['folds']:
                counts['child_models'] += len(fold['models']); counts['child_caps'] += sum(not m['fit']['optimizer_converged'] for m in fold['models'])
        rows.append({'dataset':name,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','robust','removed')} for m in ALL_METHODS}})
    assert counts['records'] == 44 and len(rows) == 15
    result = {'rows':rows,'counts':counts,
        'means':{m:{metric:float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean','noise','robust','removed')} for m in ALL_METHODS},
        'contrasts':{metric:{m:paired([r['methods']['refined'][metric]-r['methods'][m][metric] for r in rows]) for m in ALL_METHODS[1:]} for metric in ('clean','noise','robust')},
        'scope':'all15_all44_common18classifier_choices_actual_count_exposed_development'}
    clean = result['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); result['matched_holm'] = corrected
    result['development_numeric_gate'] = bool(all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0
        and result['means']['refined']['removed'] >= .1 and counts['count_mismatches'] == counts['parent_caps'] == counts['child_caps'] == 0)
    result['independent_audit_required'] = True; result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'rows'},flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((INPUT/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
