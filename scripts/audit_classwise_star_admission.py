"""Independent star knot extrema and stable-edge conditions, no classifier."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_classwise_star_admission import ROOT,FOLDER,SOURCE,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_forest_compact_transfer import equivalent


def verify_stable(lo,hi,y,i,j,wide):
    for enemy in np.flatnonzero(y!=y[i]):
        if enemy==j:continue
        maximum=0.
        for a,w in enumerate(wide):
            if w==0:continue
            knots={lo[i,a],hi[i,a]}
            for q in (lo[j,a],hi[j,a],lo[enemy,a],hi[enemy,a],(lo[j,a]+hi[j,a])/2):knots.add(float(np.clip(q,lo[i,a],hi[i,a])))
            maximum+=w*max(max(abs(t-lo[j,a]),abs(t-hi[j,a]))-max(0,lo[enemy,a]-t,t-hi[enemy,a]) for t in knots)
        assert maximum < -1e-13
    upper=sum(w*max(abs(lo[i,a]-hi[j,a]),abs(hi[i,a]-lo[j,a])) for a,w in enumerate(wide))
    assert upper<=1-1e-12+1e-14


def audit(name):
    threadpool_limits(1);path=FOLDER/(name+'.json');s=json.loads(path.read_text());counts=Counter(datasets=1)
    for rel,sha in s['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    source=json.loads((SOURCE/(name+'.json')).read_text());a=json.loads((SOURCE/'complete-audit.json').read_text())
    assert a['passed'] and a['artifact_hashes'][name]==digest(SOURCE/(name+'.json'))
    X,y=load_dataset(name);ri=np.array(s['reference_indices']);raw,labels=X[ri],y[ri]
    for block,old in zip(s['inner'],source['inner'],strict=True):
        assert block['reference_indices']==old['reference_indices'];ca=np.array(block['reference_indices'])
        A,_=preprocess_fit(raw[ca],raw[ca]);train_y=labels[ca];lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01)
        full=np.ones(A.shape[1])/np.sqrt(A.shape[1]);classes=np.unique(train_y)
        assert len(block['checks'])==len(old['stage']['certificate_checks'])
        for check,guard in zip(block['checks'],old['stage']['certificate_checks'],strict=True):
            assert check['guard_id']==guard['id'] and check['features']==guard['features'];equivalent(check['weights'],guard['weights'])
            assert check['old_accepted']==guard['accepted'] and check['majority_feasible_unchanged']==guard['certificate']['majority_feasible']
            report=check['certificate'];equivalent(report['original_class_certificate'],guard['class_certificate'])
            wide=np.zeros(A.shape[1]);wide[check['features']]=check['weights'];coefficient=full-wide;bounds=[]
            if not report['class_reports']:
                assert len(classes)==1 or np.array_equal(full,wide);bounds=[0.]*len(classes)
            else:
                assert len(report['class_reports'])==len(classes)
                for ci,(c,r) in enumerate(zip(classes,report['class_reports'],strict=True)):
                    assert r['class']==c and r['count']==sum(train_y==c)
                    assert len(r['coordinates'])==A.shape[1]
                    sources=[i for i,j in r['edges']];centers=[j for i,j in r['edges']]
                    assert len(sources)==len(set(sources)) and not set(sources)&set(centers)
                    assert sorted(sources)==r['selected_indices']
                    for i,j in r['edges']:
                        assert train_y[i]==c and train_y[j]!=c;verify_stable(lo,hi,train_y,i,j,wide)
                        if r['exactness_conditions_hold']:verify_stable(lo,hi,train_y,i,j,full)
                    if r['exactness_conditions_hold']:assert sorted(sources)==np.flatnonzero(train_y==c).tolist()
                    total=0.
                    for dim,coordinate in enumerate(r['coordinates']):
                        brute=0.
                        for center in sorted(set(centers)):
                            leaves=[i for i,j in r['edges'] if j==center]
                            knots=np.unique(np.r_[lo[center,dim],hi[center,dim],
                                np.clip(lo[leaves,dim],lo[center,dim],hi[center,dim]),
                                np.clip(hi[leaves,dim],lo[center,dim],hi[center,dim])])
                            maximum=max(sum(coefficient[dim]*(max(abs(t-lo[i,dim]),abs(t-hi[i,dim])) if coefficient[dim]>=0
                                else max(0,lo[i,dim]-t,t-hi[i,dim])) for i in leaves) for t in knots)
                            brute+=maximum;counts['independent_star_knot_replays']+=1
                        equivalent(brute,coordinate['maximum']);total+=brute
                        witness=np.array(coordinate['witness']);assert np.all(witness>=lo[:,dim]) and np.all(witness<=hi[:,dim])
                        equivalent(float(sum(coefficient[dim]*abs(witness[i]-witness[j]) for i,j in r['edges'])),brute)
                    excluded=(train_y==c)&~np.isin(np.arange(len(A)),sources)
                    remaining=float(np.array(guard['certificate']['dependency_certificate']['redistributed_object_bounds'])[excluded].sum())
                    equivalent(remaining,r['remaining_objectwise_sum']);raw_bound=(total+remaining)/sum(train_y==c)
                    equivalent(raw_bound,r['raw_signed_class_loss_bound']);star=float(np.clip(raw_bound+1e-10*(1+abs(raw_bound)),0,1))
                    equivalent(star,r['star_class_upper_bound']);bound=min(guard['class_certificate']['upper_bounds'][ci],star)
                    equivalent(bound,r['upper_bound']);bounds.append(bound);counts['independent_class_bounds']+=1
            equivalent(bounds,report['class_certificate']['upper_bounds'])
            assert report['class_certificate']['eta']==.1
            equivalent(max(bounds),report['class_certificate']['max_upper_bound'])
            assert report['class_certificate']['feasible_all_classes']==all(v<=.1 for v in bounds)
            global_bound=min(guard['certificate']['dependency_certificate']['upper_bound'],
                sum(sum(train_y==c)*bound for c,bound in zip(classes,bounds))/len(A))
            equivalent(global_bound,report['global_upper_bound'])
            if 'worst_prior_upper_bound' in report:equivalent(max(bounds),report['worst_prior_upper_bound'])
            feasible=guard['certificate']['majority_feasible'] and report['global_upper_bound']<=.1 and all(v<=.1 for v in bounds)
            assert check['new_feasible']==bool(feasible) and check['new_admission']==bool(feasible and not guard['accepted'])
            counts['guard_checks']+=1;counts['new_admissions']+=check['new_admission']
            counts['total_class_bounds']+=len(classes)
            counts['class_bounds_strictly_tightened']+=sum(np.array(bounds)<np.array(guard['class_certificate']['upper_bounds'])-1e-12)
        counts['inner_folds']+=1
        print(name,'independent class-star knot/stability/bound audit',len(block['checks']),flush=True)
    return {'dataset':name,'sha256':digest(path),'counts':dict(counts)}


def main():
    summary=json.loads((FOLDER/'complete.json').read_text());reports=[];counts=Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit,r['dataset']) for r in summary['rows']]):
            r=f.result();reports.append(r);counts.update(r['counts'])
    assert counts['datasets']==15 and counts['inner_folds']==44
    assert counts['guard_checks']==summary['counts']['guard_checks'] and counts['new_admissions']==summary['counts']['new_admissions']
    assert counts['total_class_bounds']==summary['counts']['total_class_bounds']
    assert counts['class_bounds_strictly_tightened']==summary['counts']['class_bounds_strictly_tightened']
    hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
    result={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
            'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,'classification_scored':False,
            'scope':'independent_scalar_knot_stability_and_class_mean_star_bounds_not_prediction_or_population_risk'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
