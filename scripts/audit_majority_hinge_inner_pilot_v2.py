"""Independent prototype LP, unchanged guard, geometric predictions and choice."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from run_majority_hinge_inner_pilot import ROOT,FOLDER,SOURCE,NAMES,REFINED,digest
from majority_hinge_stage import GRID
from audit_majority_hinge_fit_v2 import check_fit
from audit_predictive_target_knn_pilot import check_stage as old_check_stage
from independent_geometric_evaluation import independent_evaluate
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1);path=FOLDER/(name+'.json');saved=json.loads(path.read_text());counts=Counter(datasets=1)
    assert saved['complete'] and not saved['new_outer_evaluation_performed']
    for rel,sha in saved['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    source=json.loads((SOURCE/(name+'.json')).read_text());audit=json.loads((SOURCE/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name]==digest(SOURCE/(name+'.json'))
    assert saved['reference_indices']==source['reference_indices'];X,y=load_dataset(name);ri=np.array(saved['reference_indices']);raw,labels=X[ri],y[ri]
    for bi,(block,old) in enumerate(zip(saved['inner'],source['inner'],strict=True)):
        assert block['reference_indices']==old['reference_indices'] and block['validation_indices']==old['validation_indices']
        ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices']);assert not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]);stage=block['stage']
        old_check_stage(A,labels[ca],{'learners':[],'certificate_checks':stage['certificate_checks']},counts)
        seen=set()
        for fit in stage['learners']:
            key=(tuple(fit['features']),tuple(fit['anchor']),fit['family']);assert key not in seen;seen.add(key)
            assert fit['fit']['constrained']==(fit['family']=='majority_hinge_constrained')
            check_fit(A,labels[ca],fit,counts)
        used=set()
        for m in REFINED:
            metric=old['options'][m][source['selection']['selected'][m]]['metric']
            base={k:metric[k] for k in ('features','weights')}|{'origin':'frozen_atomic_v2_selected_metric'}
            equivalent(base,block['bases'][m]);assert len(block['options'][m])==len(GRID)
            for i,op in enumerate(block['options'][m]):
                assert op['target_index']==i;metric=op['metric'];family,t=GRID[i]
                assert metric['features']==base['features'] and metric['target_family']==family and metric['interpolation']==t
                learner=stage['learners'][metric['learner_id']];assert learner['family']==family
                equivalent(base['weights'],learner['anchor']);assert learner['features']==base['features']
                request=(1-t)*np.array(base['weights'])+t*np.array(learner['target']);equivalent(request.tolist(),metric['requested_weights'])
                if m=='proposed':
                    guard=stage['certificate_checks'][metric['guard_id']];equivalent(guard['weights'],request.tolist())
                    assert metric['guard_fallback']==(not guard['accepted']);deployed=stage['certificate_checks'][metric['deployed_guard_id']]
                    assert deployed['accepted'];actual=base['weights'] if metric['guard_fallback'] else request.tolist()
                    equivalent(actual,metric['weights']);equivalent(actual,deployed['weights'])
                else:
                    assert metric['guard_id'] is metric['deployed_guard_id'] is None and not metric['guard_fallback']
                    equivalent(request.tolist(),metric['weights'])
                rid=op['record_id'];r=block['records'][rid];used.add(rid)
                equivalent({k:metric[k] for k in ('features','weights')},{k:r['metric'][k] for k in ('features','weights')})
        assert used==set(range(len(block['records'])))
        for r in block['records']:
            equivalent(independent_evaluate(A,labels[ca],B,labels[cb],r['metric']),r['evaluation']);counts['independent_inner_predictions']+=1
        counts['inner_folds']+=1;print(name,'inner',bi+1,'independent hinge/LP/guard/vote replayed',flush=True)
    for m in REFINED:
        rows=[]
        for i in range(len(GRID)):
            scores=[b['records'][b['options'][m][i]['record_id']]['evaluation']['metrics'] for b in saved['inner']]
            rows.append({'target_index':i,'mean_BA':float(np.mean([r['clean_BA'] for r in scores])),
                'mean_logloss':float(np.mean([r['balanced_logloss'] for r in scores]))})
        chosen=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),i))
        equivalent(saved['selection'][m],{'options':rows,'selected':chosen,'selected_inner_gain_pp':100*(rows[chosen]['mean_BA']-rows[0]['mean_BA'])})
    return {'dataset':name,'counts':dict(counts),'sha256':digest(path),'selection':saved['selection']}


def main():
    summary=json.loads((FOLDER/'complete.json').read_text());reports=[];counts=Counter()
    for n in NAMES:
        r=audit(n);reports.append(r);counts.update(r['counts'])
    assert counts['datasets']==3 and counts['inner_folds']==summary['inner_folds']==9
    assert counts['target_optimizer_failures']==summary['target_optimizer_failures']
    for m in REFINED:equivalent(summary['mean_inner_gain_pp'][m],float(np.mean([r['selection'][m]['selected_inner_gain_pp'] for r in reports])))
    hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
    output={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,'new_outer_evaluation_performed':False,
        'scope':'independent_scalar_hinge_faces_numeric_LP_dual_guard_predictions_choices_not_test_or_population_risk'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
