import itertools
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from independent_positive_graph_embedding_v4 import TrueEmbedding

CASES=[
    (np.array([[.04],[.12],[.2]]),np.array([0,1,1]),[3.],[1.],0,.06),
    (np.array([[0.,.4],[.05,.95],[.7,.2],[1.,.9]]),np.array([0,0,1,1]),[1.,.2],[1.2,.8],0,.35),
    (np.array([[.1,.2],[.5,.1],[.7,.8]]),np.array([0,0,1]),[1.,.2],[0.,0.],0,.15),
    (np.array([[.1,.2],[.5,.1],[.7,.8]]),np.array([0,0,1]),[0.,0.],[1.,.2],0,.15),
    (np.array([[0.,0.],[1.,1.],[1.,1.]]),np.array([0,1,1]),[50.,0.],[0.,30.],0,.01),
    (np.array([[.5],[.5],[.5],[.5]]),np.array([8,8,-3,-3]),[1.],[.8],8,.1),
    (np.array([[0.],[.2],[.4]]),np.array([0,1,1]),[.5],[1.],0,0.),
    (np.array([[.1,.9],[.6,.4],[.8,.7]]),np.array([0,1,1]),[0.,0.],[0.,0.],0,.4),
]


@pytest.mark.parametrize('A,y,wf,wr,target,eps',CASES)
@pytest.mark.parametrize('exact',[False,True])
@pytest.mark.parametrize('positive',[False,True])
def test_true_tables_embed_with_both_targets_signed_cuts_enemy_cuts_and_saturation(A,y,wf,wr,target,eps,exact,positive):
    wf,wr=np.asarray(wf),np.asarray(wr);lo,hi=np.maximum(0.,A-eps),np.minimum(1.,A+eps);objects=[];cuts={}
    for i in range(len(A)):
        enemies=[j for j in range(len(A)) if y[j]!=y[i]]
        full_cap=min(1.,min(sum(wf[a]*max(abs(lo[i,a]-hi[j,a]),abs(hi[i,a]-lo[j,a])) for a in range(A.shape[1])) for j in enemies))
        lower={j:sum(wr[a]*max(0.,lo[i,a]-hi[j,a],lo[j,a]-hi[i,a]) for a in range(A.shape[1])) for j in enemies}
        objects.append(full_cap-min(1.,min(lower.values())));cuts[i]={j:full_cap-v for j,v in lower.items()}
    validator=TrueEmbedding(A,y,wf,wr,target,eps=eps,exact_abs=exact,positive=positive,object_upper=objects,enemy_upper=cuts)
    for corner in itertools.product((0,1),repeat=A.size):validator.table(np.where(np.asarray(corner).reshape(A.shape),hi,lo))
    rng=np.random.default_rng(31207)
    for _ in range(8):validator.table(rng.uniform(lo,hi))


def test_original_seed839_exact_attack_embeds_in_frozen_v4_positive_model():
    import json
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'results/positive-graph-independent-embedding-1791763926622800200.json').read_text())
    validator=TrueEmbedding(np.asarray(record['A']),np.asarray(record['y']),record['full'],record['reduced'],0,eps=.08,positive=True)
    actual,residual=validator.table(record['actual_z_table'])
    assert actual==pytest.approx(.022875358978938373,abs=1e-12) and residual<=1e-12
