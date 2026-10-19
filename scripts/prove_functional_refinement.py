"""Exact rational refinement identities and changed nodal penalty example."""
import json
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]


def gram(knots):
    H = sp.zeros(len(knots))
    for i in range(len(knots)-1):
        h = knots[i+1]-knots[i]
        H[i,i] += h/3; H[i+1,i+1] += h/3; H[i,i+1] += h/6; H[i+1,i] += h/6
    return H


def main():
    x = sp.symbols('x',real=True)
    coarse = [sp.Rational(0),sp.Rational(1,2),sp.Rational(1)]
    fine = [sp.Rational(i,4) for i in range(5)]
    Hc,Hf = gram(coarse),gram(fine)
    P = sp.Matrix([[1,0,0],[sp.Rational(1,2),sp.Rational(1,2),0],
        [0,1,0],[0,sp.Rational(1,2),sp.Rational(1,2)],[0,0,1]])
    assert P.T*Hf*P == Hc
    muc,muf = Hc*sp.ones(3,1),Hf*sp.ones(5,1)
    assert P.T*muf == muc
    vc,vf = sp.Matrix([t-sp.Rational(1,2) for t in coarse]),sp.Matrix([t-sp.Rational(1,2) for t in fine])
    assert P*vc == vf and (vc.T*sp.ones(3,1))[0] == (vf.T*sp.ones(5,1))[0] == 0
    old,new = (vc.T*vc)[0],(vf.T*vf)[0]
    rawc,rawf = (vc.T*Hc*vc)[0],(vf.T*Hf*vf)[0]
    assert rawc == rawf == sp.integrate((x-sp.Rational(1,2))**2,(x,0,1)) == sp.Rational(1,12)
    assert old == sp.Rational(1,2) and new == sp.Rational(5,8) and new/old == sp.Rational(5,4)
    assert new**2/old**2 == sp.Rational(25,16) and rawc**2 == sp.Rational(1,144)
    # Verify every hat Gram entry directly by exact interval integration.
    for knots,H in [(coarse,Hc),(fine,Hf)]:
        integrated = sp.zeros(len(knots))
        for a in range(len(knots)-1):
            h = knots[a+1]-knots[a]
            left,right = (knots[a+1]-x)/h,(x-knots[a])/h
            for u,f in [(a,left),(a+1,right)]:
                for v,g in [(a,left),(a+1,right)]: integrated[u,v] += sp.integrate(f*g,(x,knots[a],knots[a+1]))
        assert integrated == H
    serialize = lambda M:[[str(v) for v in row] for row in M.tolist()]
    result = {'passed':True,'coarse_gram':serialize(Hc),'fine_gram':serialize(Hf),'refinement_matrix':serialize(P),
        'identity':'P_transpose_H_fine_P_equals_H_coarse_and_P_transpose_mean_fine_equals_mean_coarse',
        'unchanged_linear_function':{'coarse_nodal_penalty':str(old),'fine_nodal_penalty':str(new),
            'nodal_unary_ratio':str(new/old),'nodal_pair_ratio':str(new**2/old**2),
            'raw_L2_unary_both':str(rawc),'raw_L2_pair_both':str(rawc**2)},
        'scope':'exact_partition_refinement_same_function_norm_counterexample_no_prediction_or_novelty_claim'}
    (ROOT/'results/functional-refinement-exact.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)


if __name__ == '__main__': main()
