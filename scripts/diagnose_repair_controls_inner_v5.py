"""Inner-only strong-control comparison; no outer score access or fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
from diagnose_repair_selector_inner_v5 import options,choose,identify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/repair-controls-inner-v5-diagnosis.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows=[];hashes={}
    for name in ('glass','heart_statlog'):
        parent=ROOT/'results/repair-relation-selector-v4'/(name+'.json')
        controls=ROOT/'results/repair-relation-controls-v4'/(name+'.json')
        if not controls.exists():continue
        p=json.loads(parent.read_text());c=json.loads(controls.read_text())
        hashes[str(parent.relative_to(ROOT))]=sha(parent);hashes[str(controls.relative_to(ROOT))]=sha(controls)
        for row in c['rows']:
            source=next(r for r in p['rows'] if r['fold']==row['fold'])['fit']
            proposed=choose(options(source['inner']),'v4_certified_tolerance')
            model_comparisons={}
            for method in row['inner'][0]['options']:
                candidates=[]
                for i in range(len(row['inner'][0]['options'][method])):
                    vals=[b['options'][method][i] for b in row['inner']]
                    if any(v['score'] is None for v in vals):continue
                    candidates.append({'index':i,'BA':float(np.mean([v['score']['clean_BA'] for v in vals])),
                        'logloss':float(np.mean([v['score']['balanced_logloss'] for v in vals])),
                        'count':float(np.mean([len(v['metric']['features']) for v in vals]))})
                if not candidates:
                    model_comparisons[method]={'available':False};continue
                chosen=max(candidates,key=lambda v:(v['BA'],-v['logloss'],-v['count'],-v['index']))
                model_comparisons[method]={'available':True,'inner_selected_config':chosen,
                    'proposed_minus_control_inner_clean_pp':100*(proposed['mean']-chosen['BA'])}
            rows.append({'dataset':name,'fold':row['fold'],'proposed_recipe':identify(proposed),'comparisons':model_comparisons})
    result={'rows':rows,'source_hashes':hashes,'script_sha256':sha(__file__),
            'outer_labels_or_scores_accessed':False,'new_fitting_or_outer_scoring_performed':False,
            'scope':'inner_selected_training_evidence_only_not_confirmatory_or_unbiased_validation'}
    for rel,v in hashes.items():assert sha(ROOT/rel)==v
    if OUT.exists():raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    for method in rows[0]['comparisons']:
        vals=[r['comparisons'][method] for r in rows if r['comparisons'][method]['available']]
        print(method,'available',len(vals),'clean deltas pp',round(float(np.mean([v['proposed_minus_control_inner_clean_pp'] for v in vals])),3) if vals else None)


if __name__=='__main__':main()
