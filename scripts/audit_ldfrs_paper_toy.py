"""Audit published Table I against explicit competing conventions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.ldfrs import local_influence

X=np.array([[.7,.9,.4,.6],[.6,.9,.3,.7],[.6,.5,.3,.5],
            [.3,.5,.2,.2],[.5,.8,.4,.6],[.4,.5,.6,.3],[.4,.4,.1,.2]])
Y=np.array([1,1,0,0,1,0,1])
PUBLISHED={
    'global_density':[2.86,2.91,2.27,2.50,2.81,2.18,2.50],
    'class_density':[2.86,2.91,2.16,2.15,2.81,2.18,1.66],
    'local_density_factor':[1.,.98,1.,1.01,1.02,.99,1.72],
    'influence':[1.,1.,.95,.86,1.,1.,0.]}
RELATION=np.array([[1,.79,.21,.01,.67,.05,0],
    [.79,1,.19,.02,.73,.03,0],[.22,.20,1,.41,.38,.26,0],
    [.02,.02,.43,1,.07,.24,0],[.67,.73,.36,.06,1,.16,0],
    [.05,.03,.50,.44,.16,1,0],[.01,.01,.23,.68,.04,.12,1]])

def main():
    rows=[]
    for factor_self in (True,False):
        density=local_influence(X,Y,k=3,density_self=True,factor_self=factor_self)
        influence=density['influence']
        for kernel in ('legacy','isotropic_gaussian','minimum_gaussian'):
            if kernel=='isotropic_gaussian':G=np.exp(-cdist(X,X)**2/(2*.25**2))
            else:
                denominator=.25 if kernel=='legacy' else 2*.25**2
                G=np.exp(-np.max((X[:,None,:]-X[None,:,:])**2,axis=2)/denominator)
            boost=(Y[:,None]==Y)&(influence[None,:]>0)&(influence[None,:]<1)
            R=np.where(boost,np.minimum(1,(1+influence[None,:])*G),influence[None,:]*G)
            for diagonal_repair in (False,True):
                relation=R.copy()
                if diagonal_repair:np.fill_diagonal(relation,1.)
                residual=np.abs(relation-RELATION)
                rows.append({'factor_self':factor_self,'kernel':kernel,
                    'diagonal_repair':diagonal_repair,
                    'density_residuals':{key:float(np.max(np.abs(density[key]-value)))
                                         for key,value in PUBLISHED.items()},
                    'relation_max_residual':float(residual.max()),
                    'relation_outside_rounding_tolerance':int((residual>.00500001).sum()),
                    'density':{key:density[key].tolist() for key in PUBLISHED},
                    'relation':relation.tolist()})
    out=Path('results/ldfrs-paper-toy-audit');out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps({'rows':rows,
        'status':'no_author_validated_convention; no_exact_toy_match'},indent=2),encoding='utf-8')
    for row in rows:
        print(row['factor_self'],row['kernel'],row['diagonal_repair'],
              row['density_residuals'],row['relation_max_residual'],
              row['relation_outside_rounding_tolerance'])

if __name__=='__main__':main()
