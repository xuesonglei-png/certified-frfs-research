"""Training-only certificate rejection audit; samples cannot prove feasibility."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import certificate,gamma


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration34';rows=[];sampled=0
    for file in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(file.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);A,_=preprocess_fit(X[tr],X[tr]);y=y[tr]
        r=next(r for r in obj['rows'] if r['method']=='protected');w=np.array(r['weights']);eta=obj['dependency_eta']
        rejected={tuple(a['features']):a for step in r['path']['trace'] for a in step['certification_attempts']
            if a['majority_feasible'] and not a['feasible']}
        rng=np.random.default_rng(701+obj['fold']);tables=[np.clip(A+rng.uniform(-.01,.01,A.shape),0,1) for _ in range(20)]
        full=[gamma(Z,y,weights=w) for Z in tables]
        for fs,attempt in rejected.items():
            c=certificate(A,y,fs,weights=w,eps=.01,method='hybrid');assert c==attempt['dependency_certificate']
            nominal=gamma(A,y,weights=w)-gamma(A,y,fs,weights=w)
            assert abs(nominal-c['nominal_loss'])<1e-12
            losses=[v-gamma(Z,y,fs,weights=w) for Z,v in zip(tables,full)];sampled+=len(losses)
            observed=max([nominal,*losses]);assert observed<=c['upper_bound']+1e-10
            category=('nominal_proves_infeasible' if nominal>eta+1e-10 else
                'sample_proves_infeasible' if observed>eta+1e-10 else 'unresolved_bound_rejection')
            zero=certificate(A,y,fs,weights=w,eps=0,method='hybrid')
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'features':list(fs),
                'input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                'category':category,'eta':eta,'nominal_loss':nominal,'sampled_max_loss':observed,
                'upper_bound':c['upper_bound'],'bound_minus_nominal':c['upper_bound']-nominal,
                'zero_radius_bound':zero['upper_bound'],'stable_fraction':c['stable_fraction'],
                'training_utility':attempt['utility']})
        print(obj['dataset'],obj['fold'],'rejections',len(rejected),flush=True)
    counts={c:sum(r['category']==c for r in rows) for c in
        ('nominal_proves_infeasible','sample_proves_infeasible','unresolved_bound_rejection')}
    result={'scope':'full_outer_training_paths_no_test_label_use_samples_not_feasibility_proof',
        'rows':rows,'counts':counts,'sampled_checks':sampled}
    (directory/'dependency-rejection-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'counts':counts,'checks':sampled,'by_dataset':{name:{c:sum(r['dataset']==name and r['category']==c for r in rows)
        for c in counts} for name in sorted(set(r['dataset'] for r in rows))}},indent=2))

if __name__=='__main__':main()
