"""Drop only identically tight own-class inequalities, preserving objective.

Frozen original and scaling implementations stay unchanged. This is a distinct
equivalent numerical representation, not new model hyperparameters.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    source = (ROOT/'src/certified_frfs/forest_robust_logloss.py').read_text(encoding='utf-8')
    old = "constraints.append(upper >= score-cp.reshape(score[np.arange(n),own],(n,1),order='C'))"
    assert source.count(old) == 1
    source = source.replace(old,"initial_ids,initial_rivals = np.where(np.arange(q)[None,:] != own[:,None])\n    constraints.append(upper[initial_ids,initial_rivals] >= score[initial_ids,initial_rivals]-score[initial_ids,own[initial_ids]])")
    source = source.replace("'class_balanced':True,'teacher_guards_enforced':False,","'class_balanced':True,'teacher_guards_enforced':False,'epigraph_layout':'rival_only_equivalent',")
    source = source.replace('Label-targeted robust log-loss envelope via shared-coordinate forest cuts.','Numerical diagnostic: equivalent rival-only nominal epigraph cuts.')
    (ROOT/'src/certified_frfs/forest_robust_logloss_rival_epigraph.py').write_text(source,encoding='utf-8')
