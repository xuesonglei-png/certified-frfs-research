"""Exact rational checks of derivative and tensor norm refinement invariance."""
import json
from pathlib import Path
import sympy as sp
from prove_functional_refinement import gram

ROOT = Path(__file__).resolve().parents[1]

def stiffness(knots):
    K = sp.zeros(len(knots))
    for j in range(len(knots)-1):
        h = knots[j+1]-knots[j]
        for u,v,value in [(j,j,1/h),(j+1,j+1,1/h),(j,j+1,-1/h),(j+1,j,-1/h)]: K[u,v] += value
    return K

if __name__ == '__main__':
    coarse = [sp.Rational(0),sp.Rational(1,2),sp.Rational(1)]
    fine = [sp.Rational(i,4) for i in range(5)]
    P = sp.Matrix([[1,0,0],[sp.Rational(1,2),sp.Rational(1,2),0],[0,1,0],[0,sp.Rational(1,2),sp.Rational(1,2)],[0,0,1]])
    Hc,Hf,Kc,Kf = gram(coarse),gram(fine),stiffness(coarse),stiffness(fine)
    alpha = sp.symbols('alpha',nonnegative=True)
    assert P.T*Kf*P == Kc and P.T*Hf*P == Hc
    Rc,Rf = Hc+alpha*Kc,Hf+alpha*Kf
    assert sp.simplify(P.T*Rf*P-Rc) == sp.zeros(3)
    tensorP = sp.kronecker_product(P,P)
    assert sp.simplify(tensorP.T*sp.kronecker_product(Rf,Rf)*tensorP-sp.kronecker_product(Rc,Rc)) == sp.zeros(9)
    result = {'passed':True,'identities':['P.T*Kfine*P=Kcoarse','P.T*(Hfine+alpha*Kfine)*P=Hcoarse+alpha*Kcoarse',
        '(P tensor P).T*(Rfine tensor Rfine)*(P tensor P)=Rcoarse tensor Rcoarse'],
        'pair_norm':'L2 + alpha*(du_L2+dv_L2) + alpha^2*du_dv_L2',
        'scope':'exact_rational_partition_instance_and_symbolic_alpha_check; general_proof_by_identical_piecewise_function_integrals; no_prediction_or_novelty_claim'}
    (ROOT/'results/sobolev-refinement-exact.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
