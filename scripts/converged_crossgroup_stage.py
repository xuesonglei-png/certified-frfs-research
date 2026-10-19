"""Uniform1000-iteration crossgroup target recipe for all weighted arms."""
from crossgroup_soft5_stage import PredictiveMetricStage as ParentStage,GRID
from crossgroup_soft5_targets import fit_crossgroup_target


class PredictiveMetricStage(ParentStage):
    def learner(self,base,family):
        fs,anchor=base['features'],base['weights'];key=(tuple(fs),tuple(anchor),family)
        if key not in self.lookup:
            target,fit=fit_crossgroup_target(self.A,self.y,fs,anchor,
                constrained=family=='crossgroup_soft5_constrained',max_iter=1000)
            fit={**fit,'neighborhood_k':5}
            self.lookup[key]=len(self.learners)
            self.learners.append({'features':fs,'anchor':anchor,'target':target.tolist(),'fit':fit,'family':family})
        return self.lookup[key]
