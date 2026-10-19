"""Shared binary atomic refinement and48-check budget for six weighted arms."""
import numpy as np
from guarded_thinned_atomic_pruning import fit_thinned_atomic_pruning

REFINED={'proposed':'proposed','unguarded_refined':'unguarded_pool','raw_repair_refined':'raw_repair_pool',
         'full_refined':'full','fgs_refined':'fgs_matched','sfrip_refined':'sfrip_matched'}


class PruningStage:
    def __init__(self,A,y):
        self.A,self.y=np.asarray(A,float),np.asarray(y);self.records=[];self.lookup={}

    def metric(self,base,variant,*,require_original=False):
        if variant not in (0,1):raise ValueError('binary refinement choice required')
        if variant==0:return {'features':list(base['features']),'weights':list(base['weights']),
            'variant':0,'pruning_id':None,'guard_reference':None}
        key=(tuple(base['features']),tuple(base['weights']))
        if key not in self.lookup:
            fs,w,fit=fit_thinned_atomic_pruning(self.A,self.y,base['features'],base['weights'],
                fraction=.5,pool=4,max_checks=48,require_original=False)
            self.lookup[key]=len(self.records);self.records.append({'initial_features':list(base['features']),
                'initial_weights':list(base['weights']),'features':fs,'weights':w.tolist(),'fit':fit})
        pid=self.lookup[key];record=self.records[pid]
        if require_original and record['fit']['guard_reference']!='original_full':
            raise RuntimeError('proposed atomic refinement lacks original-full admission')
        return {'features':record['features'].copy(),'weights':record['weights'].copy(),
            'variant':1,'pruning_id':pid,'guard_reference':record['fit']['guard_reference']}
