"""Join existing sufficient certificates without changing their semantic targets."""
from __future__ import annotations
import numpy as np

from certified_frfs.majority_metric import MajorityTerms
from signed_classwise_dependency import signed_classwise_dependency
from expanded_majority_guard_v1 import expanded_bounds


class JointAdmission:
    def __init__(self, A, y):
        self.A, self.y = np.asarray(A,float), np.asarray(y)
        self.classes = np.unique(y)
        self.protected = {}; self.dependencies = {}; self.majorities = {}

    def mask(self, reference):
        key = tuple(reference)
        if key not in self.protected:
            self.protected[key] = MajorityTerms(self.A,self.y,weights=np.asarray(reference),eps=.01).bounds(
                range(self.A.shape[1]),allowed_enemies=2)>1e-10
        return self.protected[key]

    def candidate(self, saved):
        fs, wr = saved['features'], np.asarray(saved['weights'])
        sides = {}; refs = []
        for name, ck, cl in (('original','certificate','classwise'),
                             ('learned','family_certificate','family_classwise')):
            cert = saved[ck]; base = cert['dependency_certificate']
            ref = np.asarray(base.get('full_weights',saved['weights']))
            refs.append(ref)
            old_dependency = bool(base['upper_bound'] <= .1 and saved[cl]['feasible'])
            key = tuple(fs), tuple(wr), tuple(ref)
            if key not in self.dependencies:
                if old_dependency:
                    self.dependencies[key] = {'used_signed_refinement':False,
                        'global_upper_bound':base['upper_bound'], 'class_upper_bounds':saved[cl]['bounds'],
                        'dependency_feasible':True, 'signed_report':None}
                else:
                    report = signed_classwise_dependency(self.A,self.y,fs,ref,wr,eps=.01,eta=.1,base_dependency=base)
                    glob = min(base['upper_bound'],report['global_upper_bound'])
                    cb = np.minimum(saved[cl]['bounds'],report['class_certificate']['upper_bounds']).tolist()
                    # Conservative guard padding matches the V4 nonidentity check.
                    identity = len(fs)==len(ref) and np.array_equal(wr,ref)
                    feasible = glob <= .1 and all(v+(0 if identity else 1e-10*(1+abs(v))) <= .1 for v in cb)
                    self.dependencies[key] = {'used_signed_refinement':True, 'global_upper_bound':glob,
                        'class_upper_bounds':cb, 'dependency_feasible':bool(feasible), 'signed_report':report}
            dep = self.dependencies[key]
            assert dep['global_upper_bound'] <= base['upper_bound']+1e-12
            assert np.all(np.asarray(dep['class_upper_bounds']) <= np.asarray(saved[cl]['bounds'])+1e-12)
            majority = cert['majority_feasible']; retained = cert['retained_counts']
            if not majority:
                mkey = tuple(fs),tuple(wr)
                if mkey not in self.majorities:
                    self.majorities[mkey] = expanded_bounds(self.A[:,fs],self.y,wr,eps=.01,n_friends=15)
                flags = self.majorities[mkey]>1e-10; mask = self.mask(ref)
                groups = [(self.y==c)&mask for c in self.classes]
                required = [int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
                assert required == cert['required']
                retained = [int((flags&g).sum()) for g in groups]
                majority = all(c>=r for c,r in zip(retained,required,strict=True))
            sides[name] = {'old_dependency_feasible':old_dependency,
                'old_majority_feasible':bool(cert['majority_feasible']),
                'refined_dependency':dep, 'refined_majority_feasible':bool(majority),
                'refined_retained_counts':retained, 'required':cert['required']}
        olddep = all(s['old_dependency_feasible'] for s in sides.values())
        oldmaj = all(s['old_majority_feasible'] for s in sides.values())
        dep = all(s['refined_dependency']['dependency_feasible'] for s in sides.values())
        maj = all(s['refined_majority_feasible'] for s in sides.values())
        flags = {'legacy':bool(olddep and oldmaj), 'signed_only':bool(dep and oldmaj),
                 'expanded_only':bool(olddep and maj), 'joint':bool(dep and maj)}
        assert flags['legacy'] == saved['joint_feasible']
        assert not flags['legacy'] or all(flags.values())
        assert not flags['signed_only'] or flags['joint']
        assert not flags['expanded_only'] or flags['joint']
        return {'features':fs,'weights':wr.tolist(),'original_reference':refs[0].tolist(),
                'learned_reference':refs[1].tolist(), 'arms':flags, 'sides':sides,
                'synergistic_admission':flags['joint'] and not(flags['signed_only'] or flags['expanded_only'])}
