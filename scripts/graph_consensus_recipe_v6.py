"""Training graph-consensus support proposals, keeping the original dual guards."""
import numpy as np
from certified_frfs.recent_grid_fast import fgs_paths_grid

def consensus_ranking(records,d):
    if not records:raise ValueError('empty graph registry')
    frequencies=np.zeros(d);ranks=np.zeros(d)
    for config,path in records:
        ranking=path['ranking']
        if sorted(ranking)!=list(range(d)):raise ValueError('invalid graph rank')
        frequencies[path['native']['features']]+=1
        for position,a in enumerate(ranking):ranks[a]+=position
    frequencies/=len(records);ranks/=len(records)
    ranking=sorted(range(d),key=lambda a:(-frequencies[a],ranks[a],a))
    return ranking,{'native_frequency':frequencies.tolist(),'mean_rank':ranks.tolist(),
        'configs':len(records),'aggregation':'training_only_all120_native_inclusion_then_Borda_rank'}
