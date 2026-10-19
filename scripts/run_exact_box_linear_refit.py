"""Separate exact-box run of the frozen support search and matched controls."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import hashlib
from pathlib import Path
import diagnose_robust_linear_refit as search
import compare_matched_robust_controls as matched
from certified_frfs.exact_box_linear_refit import fit_exact_box_linear

search.FOLDER=search.ROOT/'results/exact-box-linear-refit'
search.fit_robust_linear=fit_exact_box_linear
matched.FOLDER=search.ROOT/'results/exact-box-matched-controls'
matched.fit_robust_linear=fit_exact_box_linear

def run(name):
    search.run(name)
    # Matched helper reads prior folder literally; use a local adaptation
    # below for complete full attribution without modifying old artifacts.
    path=search.FOLDER/f'{name}.json';p=json.loads(path.read_text())
    additions=[Path(__file__),search.ROOT/'src/certified_frfs/exact_box_linear_refit.py',search.ROOT/'research/exact-box-linear-refit-protocol.md']
    p['fingerprints'].update({str(f.relative_to(search.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in additions})
    path.write_text(json.dumps(p,indent=2),encoding='utf-8')

def attribution(names):
    import numpy as np
    from certified_frfs.datasets import load_dataset
    from certified_frfs.validated_pool import preprocess_fit
    rows=[]
    for name in names:
        p=json.loads((search.FOLDER/f'{name}.json').read_text());t=json.loads((search.ROOT/'results/linear-retention'/f'{name}.json').read_text());X,y=load_dataset(name);scores=[]
        for index,s in enumerate(p['splits']):
            full=next(r['fit'] for r in s['trials'] if len(r['candidate']['features'])==p['d']);teacher=t['splits'][index]
            if not full['feasible']:full={'coef':teacher['teacher_coef'],'intercept':teacher['teacher_intercept']}
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);_,B=preprocess_fit(X[ri],X[vi]);classes=np.array(teacher['teacher_classes'])
            metrics=matched.metrics([B],y[vi],classes,full)
            scores.append({'proposed':s['metrics']['clean'],'full':metrics['clean']})
        rows.append({'dataset':name,'gain':float(np.mean([s['proposed']-s['full'] for s in scores])),'splits':scores})
    gain=float(np.mean([r['gain'] for r in rows]));wins=sum(r['gain']>1e-12 for r in rows)
    summary=json.loads((search.FOLDER/'complete.json').read_text());result={'rows':rows,'gain':gain,'wins':wins,'mean_removed':summary['mean_removed'],
        'gate_passed':gain>=.005 and wins>=10 and summary['mean_removed']>=.1,'scope':'exact_same_training_full_attribution_exposed_development'}
    (search.FOLDER/'complete-attribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((search.ROOT/'results/linear-retention/complete.json').read_text())['rows']];search.FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    search.summarize(names);attribution(names)
