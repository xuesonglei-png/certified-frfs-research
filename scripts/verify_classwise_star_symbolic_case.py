"""Exact SymPy example: shared-reference class certificate, no classifier."""
import json
from itertools import product
import sympy as s
from run_simplex_soft5_knn_pilot import ROOT,digest


def main():
    freeze=json.loads((ROOT/'research/classwise-star-admission-source-freeze.json').read_text())
    for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    R=s.Rational;A=[[R(0),R(3,100)],[R(1),R(0)],[R(45,100),R(4,10)]];eps=R(1,100)
    lo=[[max(s.S.Zero,v-eps) for v in row] for row in A];hi=[[min(s.S.One,v+eps) for v in row] for row in A]
    w=1/s.sqrt(2);maxima=[s.S.NegativeInfinity,s.S.NegativeInfinity];records=[]
    for corner in product([0,1],repeat=6):
        Z=[[hi[i][a] if corner[2*i+a] else lo[i][a] for a in range(2)] for i in range(3)]
        full=[];reduced=[]
        for i in range(3):
            enemy=[2] if i<2 else [0,1]
            full.append(s.Min(s.S.One,*[w*sum(abs(Z[i][a]-Z[j][a]) for a in range(2)) for j in enemy]))
            reduced.append(s.Min(s.S.One,*[s.sqrt(2)*abs(Z[i][1]-Z[j][1]) for j in enemy]))
        losses=[s.simplify(((full[0]-reduced[0])+(full[1]-reduced[1]))/2),s.simplify(full[2]-reduced[2])]
        for c in range(2):maxima[c]=s.Max(maxima[c],losses[c])
        records.append({'corner':corner,'class_losses':[str(v) for v in losses]})
    expected=[R(27,200)/s.sqrt(2),R(11,100)/s.sqrt(2)]
    assert all(s.simplify(a-b)==0 for a,b in zip(maxima,expected))
    old=R(29,200)/s.sqrt(2);eta=R(1,10);worst=s.Max(*maxima)
    assert bool(worst<eta) and bool(old>eta)
    # Exact squared comparisons eliminate numerical ambiguity at sqrt2.
    assert 27**2*2<40**2 and 29**2*2>40**2
    report={'exact_class_maxima':[str(v) for v in maxima],'exact_worst_prior_loss':str(worst),
        'old_class0_objectwise_upper':str(old),'eta':str(eta),'new_feasible_old_infeasible_exactly':True,
        'corners_independently_enumerated':len(records),'records':records,'fingerprints':freeze['fingerprints'],
        'goal_achieved':False,'classification_evaluation_performed':False,
        'scope':'exact_symbolic_three_object_FRS_uncertainty_example_not_real_data_prediction_advantage'}
    (ROOT/'results/classwise-star-symbolic-case.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v for k,v in report.items() if k not in ('records','fingerprints')},flush=True)


if __name__=='__main__':main()
