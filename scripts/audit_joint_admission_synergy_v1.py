"""Audit all four-arm flags, original masks, and numerical new admissions.

Legacy dependency values remain frozen inputs. All majority geometry is
independently replayed. Signed geometry is independently replayed for every
newly admitted metric, not for every failed refinement in the fixed pool.
"""
from collections import Counter
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_expanded_majority_admission_v1 import scalar_coordinate_endpoints,counts
from independent_joint_dependency_v1 import audit_new_signed

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/joint-admission-synergy-v1'
NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')
ARMS=('legacy','signed_only','expanded_only','joint')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(family,c):
    refs=[c[k]['dependency_certificate'].get('full_weights',c['weights']) for k in ('certificate','family_certificate')]
    return family,tuple(c['features']),tuple(c['weights']),tuple(refs[0]),tuple(refs[1])


def audit(name,hashes,artifact_hashes):
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text());assert digest(path)==artifact_hashes[name]
    assert saved['complete'] and saved['source_hashes']==hashes
    assert saved['new_classifier_predictions'] is False and saved['validation_or_outer_observations_used'] is False
    parent=json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text())
    panel=parent['rows'][0];block=panel['fit']['inner'][0]
    X,y=load_dataset(name);ii=np.asarray(panel['train_indices'])[np.asarray(block['fit_indices'])]
    A,_=preprocess_fit(X[ii],X[ii]);y=y[ii];d=A.shape[1];classes=np.unique(y)
    expected=[]
    for family,part in block['families'].items():
        unique={key(family,c):c for recipe in part['candidates'] for c in recipe['trace']}
        expected.extend((family,c) for c in unique.values())
    assert len(expected)==len(saved['rows']) and saved['d']==d
    geometries={};numeric=set();tally=Counter();new=[]
    def geometry(fs,w):
        mk=tuple(fs),tuple(w)
        if mk not in geometries:
            geometries[mk]=scalar_coordinate_endpoints(A[:,fs],y,np.asarray(w))
        return geometries[mk]
    for index,((family,c),row) in enumerate(zip(expected,saved['rows'],strict=True)):
        assert row['family']==family and row['features']==c['features'] and row['weights']==c['weights']
        fs=c['features'];wr=np.asarray(c['weights'])
        assert len(fs)==len(set(fs)) and all(0<=a<d for a in fs)
        assert wr.min()>=0 and wr.max()<=2/np.sqrt(d)+1e-9
        np.testing.assert_allclose(wr.sum(),np.sqrt(d),atol=1e-9,rtol=0)
        small,large=geometry(fs,wr);original_flags=small>1e-10;expanded_flags=large>1e-10
        assert np.all(large>=small-1e-12) and np.all(~original_flags|expanded_flags)
        olddeps=[];oldmajs=[];newdeps=[];newmajs=[]
        for side,ck,cl,rk in (('original','certificate','classwise','original_reference'),
                              ('learned','family_certificate','family_classwise','learned_reference')):
            cert=c[ck];base=cert['dependency_certificate'];ref=np.asarray(base.get('full_weights',c['weights']))
            np.testing.assert_array_equal(row[rk],ref)
            if side=='original':np.testing.assert_allclose(ref,np.ones(d)/np.sqrt(d),atol=1e-12,rtol=0)
            protected=geometry(range(d),ref)[0]>1e-10
            required=[int(np.ceil(.9*np.sum(protected&(y==q))-1e-12)) for q in classes]
            oldretained=counts(original_flags,protected,y);expandedretained=counts(expanded_flags,protected,y)
            assert cert['required']==required and cert['retained_counts']==oldretained
            oldmaj=all(a>=b for a,b in zip(oldretained,required,strict=True))
            assert oldmaj==cert['majority_feasible']
            majority=oldmaj or all(a>=b for a,b in zip(expandedretained,required,strict=True))
            sideout=row['sides'][side];dep=sideout['refined_dependency']
            olddep=base['upper_bound']<=.1 and c[cl]['feasible']
            assert sideout['old_dependency_feasible']==olddep and sideout['old_majority_feasible']==oldmaj
            assert sideout['required']==required
            assert sideout['refined_retained_counts']==(oldretained if oldmaj else expandedretained)
            assert sideout['refined_majority_feasible']==majority
            if dep['used_signed_refinement']:
                report=dep['signed_report'];assert report is not None
                # Scope is explicit: this report bounds signed class means.
                assert 'signed' in report['class_certificate']['scope']
                np.testing.assert_allclose(report['signed_object_report']['base_dependency']['upper_bound'],base['upper_bound'],atol=0,rtol=0)
                glob=min(base['upper_bound'],report['global_upper_bound'])
                cb=np.minimum(c[cl]['bounds'],report['class_certificate']['upper_bounds'])
                identity=len(fs)==len(ref) and np.array_equal(wr,ref)
                feasible=glob<=.1 and all(v+(0 if identity else 1e-10*(1+abs(v)))<=.1 for v in cb)
            else:
                assert olddep and dep['signed_report'] is None
                glob=base['upper_bound'];cb=np.asarray(c[cl]['bounds']);feasible=True
            np.testing.assert_allclose(dep['global_upper_bound'],glob,atol=1e-12,rtol=0)
            np.testing.assert_allclose(dep['class_upper_bounds'],cb,atol=1e-12,rtol=0)
            assert dep['dependency_feasible']==feasible
            assert glob<=base['upper_bound']+1e-12 and np.all(cb<=np.asarray(c[cl]['bounds'])+1e-12)
            olddeps.append(olddep);oldmajs.append(oldmaj);newdeps.append(feasible);newmajs.append(majority)
            # Replay every newly admitted numerical metric only once per reference.
            numeric_key=tuple(fs),tuple(wr),tuple(ref)
            if row['arms']['joint'] and not row['arms']['legacy'] and dep['used_signed_refinement'] and numeric_key not in numeric:
                numerical=audit_new_signed(A,y,fs,ref,wr,dep['signed_report'])
                tally.update({'numerical_'+k:v for k,v in numerical.items()});numeric.add(numeric_key)
                tally['independent_signed_star_new_guards']+=1
            tally['dual_guard_flag_checks']+=1
        arms={'legacy':all(olddeps) and all(oldmajs),
              'signed_only':all(newdeps) and all(oldmajs),
              'expanded_only':all(olddeps) and all(newmajs),
              'joint':all(newdeps) and all(newmajs)}
        assert row['arms']==arms and arms['legacy']==c['joint_feasible']
        synergy=arms['joint'] and not(arms['signed_only'] or arms['expanded_only'])
        assert row['synergistic_admission']==synergy
        tally['candidates']+=1;tally['synergistic_admissions']+=synergy
        for arm in ARMS:tally[arm]+=arms[arm]
        if arms['joint'] and not arms['legacy']:
            new.append({'row':index,'family':family,'features':fs,'signed_only':arms['signed_only'],
                        'expanded_only':arms['expanded_only'],'synergy':synergy})
    for k,v in saved['counts'].items():assert tally[k]==v,(name,k)
    minimum={arm:min((len(r['features']) for r in saved['rows'] if r['arms'][arm]),default=None) for arm in ARMS}
    assert minimum==saved['minimum_feasible_count']
    removal=(minimum['legacy']-minimum['joint'])/d
    np.testing.assert_allclose(saved['extra_removal'],removal,atol=1e-12,rtol=0)
    print(name,'all flags/masks and new signed-star geometry passed',dict(tally),flush=True)
    return {'dataset':name,'passed':True,'checks':dict(tally),'new_admissions':new,
            'minimum_feasible_count':minimum,'extra_removal':removal,'artifact_sha256':digest(path)}


def main():
    threadpool_limits(1);out=FOLDER/'independent-combination-audit-v1.json';assert not out.exists()
    summary=json.loads((FOLDER/'summary.json').read_text());hashes=json.loads((FOLDER/'source-freeze.json').read_text())['hashes']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    start=time.perf_counter();reports=[audit(n,hashes,summary['artifact_hashes']) for n in NAMES];tally=Counter()
    for r in reports:tally.update(r['checks'])
    for k,v in summary['counts'].items():assert tally[k]==v,k
    improved=sum(r['minimum_feasible_count']['joint']<r['minimum_feasible_count']['legacy'] for r in reports)
    removal=float(np.mean([r['extra_removal'] for r in reports]));gate=improved>=3 and removal>=.05
    assert summary['improved_datasets']==improved and summary['resource_gate']==gate
    np.testing.assert_allclose(summary['mean_extra_removal'],removal,atol=1e-12,rtol=0)
    files=[Path(__file__),ROOT/'scripts/independent_joint_dependency_v1.py',
           ROOT/'scripts/independent_signed_classwise.py',ROOT/'scripts/audit_expanded_majority_admission_v1.py']
    result={'passed':True,'checks':dict(tally),'dataset_rows':reports,'seconds':time.perf_counter()-start,
            'improved_datasets':improved,'mean_extra_removal':removal,'resource_gate':gate,
            'auditor_hashes':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
            'new_classifier_predictions':False,'new_optimization_or_attacks':False,'goal_achieved':False,
            'scope':'all_four_arm_flags_and_majority_geometry_new_admission_signed_star_geometry',
            'limits':['Legacy dependency certificates are frozen inputs, not rederived here.',
                      'Signed/star geometry independently replayed for new admissions only.',
                      'Class signed mean guarantee does not preserve a mean-positive-object-loss constraint.',
                      'Production preprocessing is reused; ordinary floating point is not formal verification.',
                      'Training pool admission or dimensional savings do not establish prediction advantage.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='dataset_rows'},indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        (FOLDER/('independent-audit-failure-'+str(time.time_ns())+'.json')).write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
        raise
