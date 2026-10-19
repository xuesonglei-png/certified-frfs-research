"""Declared broad FGS numerical grid, keeping the previous nine configs.

Paper reports beta .1..1 step .1, xi .1..2 step .2, whose stated endpoints
do not land exactly on2. Include the odd-tenth progression and explicit1/2
from the existing grid. This is a declared superset, not author defaults.
"""
from fractions import Fraction
from certified_frfs.recent_grid_fast import fgs_paths_grid

BETAS=tuple(float(Fraction(i,10)) for i in range(1,11))
XIS=tuple(float(v) for v in sorted({Fraction(i,10) for i in range(1,20,2)}|{Fraction(1),Fraction(2)}))
CONFIGS=tuple({'beta':b,'xi':x} for b in BETAS for x in XIS)
LEGACY=tuple({'beta':b,'xi':x} for b in (.6,.8,.9) for x in (.5,1.,2.))


def expanded_paths(A,y):
    # The original fast engine retains complement tables perxi; batching
    # one xi at a time limits peak memory without changing any arithmetic.
    lookup={}
    for xi in XIS:
        batch=[{'beta':b,'xi':xi} for b in BETAS]
        paths=fgs_paths_grid(A,y,batch)
        for config,path in zip(batch,paths,strict=True):
            lookup[(config['beta'],config['xi'])]=path
    return [{'index':i,'config':dict(c),'path':lookup[(c['beta'],c['xi'])]} for i,c in enumerate(CONFIGS)]
