"""Full refit of each certified deletion path and independent score rebuild."""
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_certified_cap_backward import ROOT, FOLDER, fingerprints
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.certified_cap_backward import certified_cap_backward
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1); counts = {'path_refits':0, 'accepted_deletions':0, 'checked_candidates':0, 'scores':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'): continue
        p = json.loads(path.read_text()); name = p['dataset']; assert p['fingerprints'] == fingerprints(name)
        prior = json.loads((ROOT/'results/capped-sparse-extension'/f'{name}.json').read_text()); X,y = load_dataset(name)
        for index,s in enumerate(p['splits']):
            old = prior['splits'][index]; ri,vi = np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices'] and set(ri).isdisjoint(vi)
            fit = certified_cap_backward(X[ri], y[ri], old['fit'])
            assert fit == s['fit']
            A,B = scaling(X[ri],X[vi]); c=fit['selected']
            metrics=validation_scores(A,y[ri],B,y[vi],c['features'],np.array(c['weights']),144+index,joint=True)
            assert all(abs(metrics[k]-s['metrics'][k])<1e-12 for k in metrics)
            counts['path_refits']+=1; counts['accepted_deletions']+=len(fit['trace'])-1
            counts['checked_candidates']+=len(fit['evaluated']); counts['scores']+=2
        print(name,'certified deletion audit complete',flush=True)
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/certified-cap-backward-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_certified_cap_backward.py','scripts/audit_certified_cap_backward.py','research/certified-cap-backward-protocol.md'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
