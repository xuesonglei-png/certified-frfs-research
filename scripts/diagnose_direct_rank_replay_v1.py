"""Read-only diagnosis of scalar-versus-batched ranks; no acceptance relaxation."""
import json
from pathlib import Path
import numpy as np
from audit_fixed_face_majority_milp import rank_report
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_simplex_soft5_knn_pilot import ROOT,digest


def main():
    path=ROOT/'results/direct-majority-frs-development/transfusion.json';saved=json.loads(path.read_text())
    X,y=load_dataset('transfusion');ri=np.asarray(saved['reference_indices']);ca=np.asarray(saved['training_indices'])
    A,_=preprocess_fit(X[ri][ca],X[ri][ca]);labels=y[ri][ca];out={'source_sha256':digest(path),'arms':{},'classification_superiority_established':False}
    for arm,fit in saved['fits'].items():
        fs=fit['features'];mass=fit['mass'];point=np.asarray(fit['seed']);fresh=rank_report(A,labels,fs,point,fit['margin']);old=fit['seed_report']
        mismatches=[]
        for a,b in zip(fresh['rows'],old['rows'],strict=True):
            i=a['query'];dist=lambda j:float(sum(abs(A[j,feature]*point[k]/mass-A[i,feature]*point[k]/mass) for k,feature in enumerate(fs)))
            for key in ('three_friends','two_allowed_enemies'):
                if a[key]!=b[key]:
                    av,bv=[dist(j) for j in a[key]],[dist(j) for j in b[key]]
                    mismatches.append({'query':i,'kind':key,'scalar_indices':a[key],'stored_indices':b[key],
                        'scalar_distances':av,'stored_distances':bv,'max_paired_distance_difference':max(abs(np.asarray(av)-np.asarray(bv)),default=0.)})
        qdiff=[a['query'] for a,b in zip(fresh['rows'],old['rows'],strict=True) if a['qualified']!=b['qualified']]
        out['arms'][arm]={'identity_mismatches':mismatches,'qualified_query_mismatches':qdiff,
            'scalar_rate':fresh['balanced_qualified_rate'],'stored_rate':old['balanced_qualified_rate'],
            'actual_accepted_descent':fit['accepted_descent']}
    target=ROOT/'research/direct-majority-rank-replay-v1-diagnosis.json';assert not target.exists()
    target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({a:{'identities_changed':len(v['identity_mismatches']),
        'max_distance_discrepancy':max((r['max_paired_distance_difference'] for r in v['identity_mismatches']),default=0.),
        'qualified_query_mismatches':v['qualified_query_mismatches'],'scalar_rate':v['scalar_rate'],'stored_rate':v['stored_rate'],
        'accepted_descent':v['actual_accepted_descent']} for a,v in out['arms'].items()},indent=2),flush=True)


if __name__=='__main__':main()
