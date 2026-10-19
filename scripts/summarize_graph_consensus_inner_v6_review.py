"""Independent registry/ranking/selection summary for the fixed six-task v6 test.

No prediction or fitting occurs here; all comparisons use training-inner score
records. The fixed gate is applied to dataset-equal three-panel mean deltas.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/graph-consensus-inner-v6'
NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')
FAMILIES=('uniform_sqrt','learned_majority','half_local_soft')
OUT=ROOT/'research/graph-consensus-inner-v6-reviewed-summary.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows=[];checks=Counter();hashes={}
    for name in NAMES:
        p=FOLDER/(name+'.json')
        saved=json.loads(p.read_text());assert saved['complete'] and not saved['new_outer_scoring']
        parent=json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text())
        controls=json.loads((ROOT/'results/repair-relation-controls-v4'/(name+'.json')).read_text())
        hashes[str(p.relative_to(ROOT))]=sha(p)
        for panel,previous,control in zip(saved['panels'],parent['rows'],controls['rows'],strict=True):
            assert panel['outer_training_fold']==previous['fold']==control['fold']
            assert len(panel['inner'])==len(previous['fit']['inner'])==len(control['inner'])
            for b,old,cb in zip(panel['inner'],previous['fit']['inner'],control['inner'],strict=True):
                assert b['fit_indices']==old['fit_indices']==cb['fit_indices']
                assert b['validation_indices']==old['validation_indices']==cb['validation_indices']
                assert not set(b['fit_indices']) & set(b['validation_indices'])
                registry=cb['raw_paths']['fgs'];assert len(registry)==120
                frequencies=[sum(a in path['native']['features'] for config,path in registry)/120 for a in range(saved['d'])]
                borda=[sum(path['ranking'].index(a) for config,path in registry)/120 for a in range(saved['d'])]
                rank=sorted(range(saved['d']),key=lambda a:(-frequencies[a],borda[a],a))
                assert rank==b['consensus_ranking']
                np.testing.assert_allclose(frequencies,b['consensus']['native_frequency'],atol=1e-12)
                np.testing.assert_allclose(borda,b['consensus']['mean_rank'],atol=1e-12)
                checks['independent_consensus_rankings']+=1
                for family in FAMILIES:
                    np.testing.assert_array_equal(b['families'][family]['reference_weights'],old['families'][family]['reference_weights'])
                    for candidate in b['families'][family]['candidates']:
                        model=candidate['model'];trace=candidate['trace']
                        requested=max((saved['d']+1)//2,int(np.ceil(saved['d']*candidate['fraction'])))
                        assert model['requested_count']==requested
                        assert model['restored_count']==len(model['features'])-requested
                        assert model['joint_feasible'] and trace[-1]['joint_feasible']
                        for index,c in enumerate(trace):
                            fallback=model['uniform_fallback'] and index==len(trace)-1
                            expected=sorted(rank[:min(requested+index,saved['d'])])
                            assert c['features']==expected
                            if index<len(trace)-1:assert not c['joint_feasible']
                            if fallback:np.testing.assert_allclose(c['weights'],np.ones(saved['d'])/np.sqrt(saved['d']),atol=1e-12)
                        checks['restoration_trace_checks']+=1
                        checks['uniform_fallbacks']+=model['uniform_fallback']
                        checks['compressed_candidates']+=len(model['features'])<saved['d']
            opts=[]
            for family in FAMILIES:
                for index in range(4):
                    cc=[b['families'][family]['candidates'][index] for b in panel['inner']]
                    opts.append({'family':family,'recipe_index':index,'mean_BA':float(np.mean([c['score']['clean_BA'] for c in cc])),
                        'mean_certified_BA':float(np.mean([c['score']['query_certified_BA'] for c in cc])),
                        'mean_count':float(np.mean([len(c['model']['features']) for c in cc]))})
            best=max(opts,key=lambda r:(round(r['mean_BA'],12),round(r['mean_certified_BA'],12),-r['mean_count'],-FAMILIES.index(r['family']),-r['recipe_index']))
            assert best['family']==panel['selected_training_only']['family'] and best['recipe_index']==panel['selected_training_only']['recipe_index']
            old_best=max(o['mean_BA'] for o in previous['fit']['options'])
            anchor=np.mean([b['families']['uniform_sqrt']['candidates'][3]['score']['clean_BA'] for b in previous['fit']['inner']])
            delta=100*(best['mean_BA']-old_best);full_delta=100*(best['mean_BA']-anchor)
            np.testing.assert_allclose([delta,full_delta],[panel['best_inner_delta_previous_pool_pp'],panel['best_inner_delta_full_anchor_pp']],atol=1e-12)
            best_cert_old=max(o['mean_certified_BA'] for o in previous['fit']['options'])
            cert_anchor=np.mean([b['families']['uniform_sqrt']['candidates'][3]['score']['query_certified_BA'] for b in previous['fit']['inner']])
            rows.append({'dataset':name,'panel':panel['outer_training_fold'],'selected':{'family':best['family'],'recipe_index':best['recipe_index']},
                'inner_clean_BA':best['mean_BA'],'inner_certified_BA':best['mean_certified_BA'],'mean_count':best['mean_count'],
                'removed':1-best['mean_count']/saved['d'],'clean_delta_previous_pool_pp':delta,'clean_delta_full_anchor_pp':full_delta,
                'certified_delta_best_previous_certified_pp':100*(best['mean_certified_BA']-best_cert_old),
                'certified_delta_full_anchor_pp':100*(best['mean_certified_BA']-cert_anchor)})
            checks['selection_and_delta_replays']+=1
    dataset_rows=[]
    for name in NAMES:
        group=[r for r in rows if r['dataset']==name]
        assert len(group)==3
        means={k:float(np.mean([r[k] for r in group])) for k in ('inner_clean_BA','inner_certified_BA','removed','mean_count','clean_delta_previous_pool_pp','clean_delta_full_anchor_pp','certified_delta_best_previous_certified_pp','certified_delta_full_anchor_pp')}
        dataset_rows.append({'dataset':name,**means,'passes_.5pp_inner_gain':means['clean_delta_previous_pool_pp']>=.5-1e-12})
    wins=sum(r['passes_.5pp_inner_gain'] for r in dataset_rows)
    result={'dataset_rows':dataset_rows,'panel_rows':rows,'checks':dict(checks),'source_hashes':hashes,'script_sha256':sha(__file__),
        'dataset_equal_mean_clean_delta_previous_pool_pp':float(np.mean([r['clean_delta_previous_pool_pp'] for r in dataset_rows])),
        'dataset_equal_mean_clean_delta_full_anchor_pp':float(np.mean([r['clean_delta_full_anchor_pp'] for r in dataset_rows])),
        'passes_.5pp_datasets':wins,'predeclared_4_of_6_gate_passed':wins>=4,'new_outer_scoring':False,'goal_achieved':False,
        'scope':'training_inner_winner_scores_optimistic_mechanism_screen_not_generalization_or_significance',
        'review_limits':['Cached FGS120 numeric paths and relation fits inherited, not independently refitted.','Candidate finite guards and predictions replayed in frozen runner; this summary replays ranks, trace order and score aggregates only.','Frozen v6 manifest omits imported auditor files; current hash record added below for review traceability.']}
    result['imported_auditor_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/audit_selective_support_selector_v3.py',ROOT/'scripts/audit_relation_support_selector_v1.py',ROOT/'scripts/audit_majority_bound_prediction_v1.py']}
    for rel,expected in hashes.items():assert sha(ROOT/rel)==expected
    if OUT.exists():raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('panel_rows','source_hashes')},indent=2))


if __name__=='__main__':main()
