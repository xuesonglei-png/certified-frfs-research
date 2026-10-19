"""ALL15 train-only fixed-anchor rank/density diagnostic, no new fitting.

Compare leave-one-out full reference5 versus crossgroup reference5. This
measures a conditional training geometry mismatch, never a test-risk cause.
All same-label duplicate/reference distance conventions are explicitly retained.
"""
import json
import time
from collections import Counter
from pathlib import Path
import ijson
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from continue_soft_topk_knn_pilot import verify
from run_simplex_soft5_knn_pilot import ROOT,digest
from ensemble_vote_objective_v1 import SEEDS
from independent_ensemble_vote_v1 import grouping

SOURCE=ROOT/'results/frs-joint-zero-face'
FREEZE=ROOT/'research/crossgroup-reference-density-v1-source-freeze.json'
OUT=ROOT/'research/crossgroup-reference-density-v1.json'


def field(path,prefix):
    with path.open('rb') as stream:return next(ijson.items(stream,prefix,use_float=True))


def coordinates(A,fs,w):
    # Accumulate weighted stored-coordinate distance independently of cdist.
    out=np.zeros((len(A),len(A)))
    for a,weight in zip(fs,w,strict=True):
        out+=abs(A[:,a,None]*weight-A[None,:,a]*weight)
    return out


def neighbors(D,allowed):
    valid=allowed.sum(axis=1)>=5
    rows=np.flatnonzero(valid)
    ranked=np.argsort(np.where(allowed[rows],D[rows],np.inf),axis=1,kind='stable')
    ids=ranked[:,:5];radius=np.take_along_axis(D[rows],ids[:,-1:],axis=1)[:,0]
    gap=np.full(len(rows),np.inf)
    has_six=allowed[rows].sum(axis=1)>5
    gap[has_six]=D[rows[has_six],ranked[has_six,5]]-radius[has_six]
    return rows,ids,radius,gap


def direct_counts(labels,ids):
    classes=sorted(set(labels.tolist()))
    return np.array([[sum(labels[j]==c for j in row) for c in classes] for row in ids])


def balanced(values,labels):
    return float(np.mean([np.mean(np.asarray(values)[labels==c]) for c in np.unique(labels)]))


def run(name,sha,counts):
    path=SOURCE/(name+'.json');assert digest(path)==sha
    ri=np.asarray(field(path,'reference_indices'));ca=np.asarray(field(path,'training_indices'))
    fs=field(path,'original_support');weights=np.asarray(field(path,'original_anchor'))
    X,y=load_dataset(name);raw=X[ri][ca];labels=y[ri][ca]
    # A single TRAINING row is a discarded dummy query for this preprocessing API.
    # No validation indices, query features or query labels enter this diagnosis.
    A,_=preprocess_fit(raw,raw[:1]);mass=np.sqrt(A.shape[1]);w=weights/mass
    D=cdist(A[:,fs]*w,A[:,fs]*w,'cityblock');independent=coordinates(A,fs,w)
    np.testing.assert_allclose(D,independent,atol=2e-14,rtol=1e-12)
    full_allowed=~np.eye(len(A),dtype=bool)
    rows,nn,radius,gap=neighbors(D,full_allowed);assert len(rows)==len(A)
    independent_rows,independent_nn,_,_=neighbors(independent,full_allowed)
    stable=gap>1e-10
    for i in np.flatnonzero(stable):assert set(nn[i])==set(independent_nn[i])
    counts['independent_distance_coordinates']+=len(fs)*len(A)**2
    counts['full_stable_neighbor_sets_replayed']+=int(stable.sum())
    classes=np.unique(labels);own=np.searchsorted(classes,labels)
    votes=np.column_stack([(labels[nn]==c).sum(axis=1) for c in classes])
    np.testing.assert_array_equal(votes,direct_counts(labels,nn))
    rivals=votes.copy();rivals[np.arange(len(A)),own]=-1
    strict=votes[np.arange(len(A)),own]>rivals.max(axis=1)
    tied=votes[np.arange(len(A)),own]==rivals.max(axis=1)
    records=[]
    for seed in SEEDS:
        group=grouping(labels,seed);allowed=group[:,None]!=group[None,:]
        valid,subset_nn,subset_radius,subset_gap=neighbors(D,allowed)
        independent_valid,independent_subset_nn,_,_=neighbors(independent,allowed)
        assert valid.tolist()==independent_valid.tolist()
        subset_stable=subset_gap>1e-10
        for i in np.flatnonzero(subset_stable):assert set(subset_nn[i])==set(independent_subset_nn[i])
        counts['subset_stable_neighbor_sets_replayed']+=int(subset_stable.sum())
        subset_votes=np.column_stack([(labels[subset_nn]==c).sum(axis=1) for c in classes])
        np.testing.assert_array_equal(subset_votes,direct_counts(labels,subset_nn))
        rivals=subset_votes.copy();rivals[np.arange(len(valid)),own[valid]]=-1
        subset_strict=subset_votes[np.arange(len(valid)),own[valid]]>rivals.max(axis=1)
        subset_tied=subset_votes[np.arange(len(valid)),own[valid]]==rivals.max(axis=1)
        overlap=np.array([len(set(nn[q])&set(subset_nn[i])) for i,q in enumerate(valid)])
        eligible_full_count=np.array([int(allowed[q,nn[q]].sum()) for q in valid])
        full_radius=radius[valid];positive=full_radius>1e-12
        records.append({'seed':seed,'eligible_queries':len(valid),
            'mean_available_reference_fraction':float(np.mean(allowed[valid].sum(axis=1)/(len(A)-1))),
            'mean_full5_references_still_allowed':float(np.mean(eligible_full_count)),
            'mean_shared_neighbor_count':float(np.mean(overlap)),
            'mean_new_farther_neighbors':float(np.mean(5-overlap)),
            'mean_radius_ratio_positive_full_radius':float(np.mean(subset_radius[positive]/full_radius[positive])) if positive.any() else None,
            'zero_full_radius_queries':int((~positive).sum()),
            'full_strict_true_plurality_fraction':balanced(strict[valid],labels[valid]),
            'subset_strict_true_plurality_fraction':balanced(subset_strict,labels[valid]),
            'full_true_vote_tie_fraction':balanced(tied[valid],labels[valid]),
            'subset_true_vote_tie_fraction':balanced(subset_tied,labels[valid]),
            'strict_correctness_status_changed_fraction':balanced(strict[valid]!=subset_strict,labels[valid]),
            'full5_boundary_ties':int((gap[valid]<=1e-10).sum()),
            'subset5_boundary_ties':int((subset_gap<=1e-10).sum()),
            'stable_subset_neighbor_sets_independently_replayed':int(subset_stable.sum())})
    counts['datasets']+=1;counts['partitions']+=len(records)
    print(name,'density/rank diagnostic independently checked',flush=True)
    return {'dataset':name,'training_objects':len(A),'features':fs,'source_sha256':sha,'partitions':records}


def main():
    threadpool_limits(1);assert not FREEZE.exists() and not OUT.exists()
    audit=verify(SOURCE);assert len(audit['artifact_hashes'])==15
    paths=[p for folder in ('src','scripts') for p in (ROOT/folder).rglob('*.py')]
    paths += [ROOT/'research/crossgroup-reference-density-v1-protocol.md',SOURCE/'complete.json',SOURCE/'complete-audit.json']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,
        'source_artifact_hashes':audit['artifact_hashes'],'scope':'ALL15_train_only_fixed_anchor_three_partition_rank_density'},indent=2),encoding='utf-8')
    counts=Counter();rows=[run(n,s,counts) for n,s in sorted(audit['artifact_hashes'].items())]
    numeric=('mean_available_reference_fraction','mean_full5_references_still_allowed','mean_shared_neighbor_count',
        'mean_new_farther_neighbors','mean_radius_ratio_positive_full_radius','full_strict_true_plurality_fraction',
        'subset_strict_true_plurality_fraction','strict_correctness_status_changed_fraction')
    means={key:float(np.mean([np.mean([r[key] for r in row['partitions'] if r[key] is not None]) for row in rows])) for key in numeric}
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    out={'passed_independent_coordinate_and_stable_rank_replay':True,'rows':rows,'means':means,'counts':dict(counts),
        'freeze_sha256':digest(FREEZE),'source_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'no_validation_rows_or_labels_used':True,'new_fitting_performed':False,'endpoint_selected':False,
        'classification_superiority_established':False,'goal_achieved':False,
        'limits':['Training fixed-anchor conditional rank mismatch is not proof of a test-risk cause.',
            'Rows near exact5NN boundaries are reported and not claimed to have uniquely replayed IDs.',
            'Strict true plurality and true vote ties are separated; tie decisions are not scored.',
            'Three reference partitions remain enclosing-training-fold geometry, not fully cross-fitted preprocessing.',
            'No new classifier, test superiority, loss calibration or independent confirmation.']}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'means':means,'counts':dict(counts)},indent=2),flush=True)


if __name__=='__main__':main()
