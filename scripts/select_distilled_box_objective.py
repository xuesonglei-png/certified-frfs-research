"""Training-only support selection over immutable certified deployed fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from compare_matched_robust_controls import ROOT, metrics
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER = ROOT / 'results/distilled-objective-selection'

def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    prior = ROOT / 'results/distilled-box-linear-refit'
    summary = json.loads((prior / 'complete.json').read_text()); rows = []
    for row in summary['rows']:
        name = row['dataset']; target = FOLDER / f'{name}.json'
        if target.exists(): raise FileExistsError(target)
        p = json.loads((prior / f'{name}.json').read_text())
        teacherpath = ROOT / 'results/linear-retention' / f'{name}.json'
        teacher = json.loads(teacherpath.read_text()); X,y = load_dataset(name)
        records = []
        for i,s in enumerate(p['splits']):
            feasible = [(j,r) for j,r in enumerate(s['trials']) if r['fit']['feasible']]
            j,r = min(feasible,key=lambda z:(z[1]['fit']['objective']+.01*len(z[1]['candidate']['features'])/p['d'],len(z[1]['candidate']['features']),z[0]))
            ri,vi = np.array(s['reference_indices']),np.array(s['validation_indices'])
            _,B = preprocess_fit(X[ri],X[vi]); rng = np.random.default_rng(144+i)
            queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            classes = np.array(teacher['splits'][i]['teacher_classes'])
            score = metrics(queries,y[vi],classes,r['fit'])
            full = next(z for z in s['trials'] if len(z['candidate']['features'])==p['d'])
            assert full['fit']['feasible']
            control = metrics(queries,y[vi],classes,full['fit'])
            records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'selected_index':j,'selected':r,
                'selection_value':r['fit']['objective']+.01*len(r['candidate']['features'])/p['d'],
                'metrics':score,'full_metrics':control})
        files = [Path(__file__),ROOT / 'research/distilled-selection-protocol.md',prior / f'{name}.json',teacherpath,ROOT / 'data/processed' / f'{name}.npz']
        target.write_text(json.dumps({'dataset':name,'d':p['d'],'splits':records,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')
        clean = float(np.mean([s['metrics']['clean'] for s in records])); fullclean = float(np.mean([s['full_metrics']['clean'] for s in records]))
        rows.append({'dataset':name,'clean':clean,'full':fullclean,'gain':clean-fullclean,
                     'removed':float(np.mean([1-len(s['selected']['candidate']['features'])/p['d'] for s in records]))})
    result = {'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_gain_vs_matched_full':float(np.mean([r['gain'] for r in rows])),
              'mean_removed':float(np.mean([r['removed'] for r in rows])),'wins':sum(r['gain']>1e-12 for r in rows),'scope':'reference_objective_selection_exposed_development_not_confirmation'}
    (FOLDER / 'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__': main()
