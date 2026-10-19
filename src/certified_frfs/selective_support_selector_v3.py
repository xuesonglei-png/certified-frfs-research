"""Fold-local recipe search with shrunk relations and unchanged dual guards.

Recipes are fixed before training.  A recipe is a learning procedure, not an
absolute feature set shared across differently fitted inner models.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .relation_support_selector_v1 import _candidate, _fit_relation, _ranking
from .relation_support_selector_v1 import weighted_knn
from .knn_query import certify_knn_queries
from .validated_pool import preprocess_fit

FAMILIES = ("uniform_sqrt", "half_majority", "learned_majority")


def query_scores(A,y,B,truth,model):
    score=weighted_knn(A,y,B,truth,model["features"],model["weights"])
    embedded=np.zeros(A.shape[1]);embedded[model["features"]]=model["weights"]
    cert=certify_knn_queries(A,y,B,np.asarray(score["predictions"]),model["features"],
                             weights=embedded,eps=.01,k=5,allowed_enemies=2)
    flags=np.asarray(cert["certified"]); correct=np.asarray(score["predictions"])==truth
    groups=[np.asarray(truth)==c for c in np.unique(truth)]
    covered=[int(np.sum(flags & g)) for g in groups]
    good=[int(np.sum(flags & correct & g)) for g in groups]
    score.update({"query_certified_BA":float(np.mean([np.mean((flags & correct)[g]) for g in groups])),
                  "query_coverage":float(flags.mean()),
                  "balanced_query_coverage":float(np.mean([flags[g].mean() for g in groups])),
                  "selective_accuracy":float(correct[flags].mean()) if np.any(flags) else None,
                  "class_certified_counts":covered,"class_certified_correct_counts":good,
                  "class_query_counts":[int(g.sum()) for g in groups],"query_certificate":cert})
    return score


def recipes(d):
    counts = sorted({max(int(np.ceil(d/2)),int(np.ceil(d*f))) for f in (.5,.7,.85,1.)})
    result=[]
    for count in counts:
        result.append({"count":count,"remove_rank":None,"add_rank":None})
        # Fixed local pool: replace either of the two boundary coordinates
        # with either of the next two.  The recipe pool uses dimension only.
        for remove in range(max(0,count-2),count):
            for add in range(count,min(d,count+2)):
                result.append({"count":count,"remove_rank":remove,"add_rank":add})
    return result


def support(ranking, recipe):
    chosen=set(ranking[:recipe["count"]])
    if recipe["remove_rank"] is not None:
        chosen.remove(ranking[recipe["remove_rank"]])
        chosen.add(ranking[recipe["add_rank"]])
    return sorted(chosen)


def relations(A,y):
    uniform=_fit_relation(A,y,"uniform_sqrt")
    learned=_fit_relation(A,y,"learned_majority")
    return {"uniform_sqrt":uniform,"half_majority":(uniform+learned)/2,
            "learned_majority":learned}


def summarize_options(blocks, grid, *, clean_tolerance=.005):
    rows=[]
    for family in FAMILIES:
        for index,recipe in enumerate(grid):
            candidates=[b["families"][family]["candidates"][index] for b in blocks]
            if not all(c["score"] is not None for c in candidates):continue
            rows.append({"family":family,"recipe_index":index,"recipe":recipe,
                         "mean_BA":float(np.mean([c["score"]["clean_BA"] for c in candidates])),
                         "mean_certified_BA":float(np.mean([c["score"]["query_certified_BA"] for c in candidates])),
                         "mean_coverage":float(np.mean([c["score"]["query_coverage"] for c in candidates]))})
    if not rows:raise RuntimeError("no recipe passes all inner certificates")
    best=max(r["mean_BA"] for r in rows)
    eligible=[r for r in rows if r["mean_BA"]>=best-clean_tolerance-1e-12]
    selected=max(eligible,key=lambda r:(r["mean_certified_BA"],r["mean_coverage"],
                                        r["mean_BA"],-r["recipe"]["count"],
                                        -FAMILIES.index(r["family"]),-r["recipe_index"]))
    return rows,selected


def fit_outer(X,y,*,seed=20261011,inner_splits=3):
    X,y=np.asarray(X,float),np.asarray(y)
    if len(np.unique(y))<2 or np.unique(y,return_counts=True)[1].min()<inner_splits:
        raise ValueError("insufficient class repetitions")
    grid=recipes(X.shape[1]); blocks=[]
    splits=StratifiedKFold(inner_splits,shuffle=True,random_state=seed)
    for fold,(ii,jj) in enumerate(splits.split(X,y)):
        A,B=preprocess_fit(X[ii],X[jj]); fitted=relations(A,y[ii]); families={}
        for family in FAMILIES:
            w=fitted[family]; ranking=_ranking(A,y[ii],w,family,seed+fold)
            candidates=[]
            for index,recipe in enumerate(grid):
                fs=support(ranking,recipe)
                c=_candidate(A,y[ii],w,fs,len(fs))
                score=query_scores(A,y[ii],B,y[jj],c) if c["joint_feasible"] else None
                candidates.append({"recipe_index":index,"candidate":c,"score":score})
            families[family]={"reference_weights":w.tolist(),"ranking":ranking,"candidates":candidates}
        blocks.append({"fold":fold,"fit_indices":ii.tolist(),"validation_indices":jj.tolist(),"families":families})
    options,chosen=summarize_options(blocks,grid)
    A,_=preprocess_fit(X,X); fitted=relations(A,y)
    w=fitted[chosen["family"]]
    ranking=_ranking(A,y,w,chosen["family"],seed+100)
    fs=support(ranking,chosen["recipe"])
    attempt=_candidate(A,y,w,fs,len(fs))
    fallback=not attempt["joint_feasible"]
    if fallback:
        uniform=fitted["uniform_sqrt"]
        selected=_candidate(A,y,uniform,list(range(A.shape[1])),A.shape[1])
    else:selected=dict(attempt)
    selected.update({"family":"uniform_sqrt" if fallback else chosen["family"],
                     "outer_refit_fallback":fallback,"requested_count":chosen["recipe"]["count"],
                     "realized_count":len(selected["features"])})
    assert selected["joint_feasible"]
    return {"selected":selected,"refit_attempt":attempt,"chosen":chosen,"options":options,
            "inner":blocks,"recipes":grid,"families":list(FAMILIES),
            "refit_reference_weights":w.tolist(),"refit_ranking":ranking,
            "seed":seed,"inner_splits":inner_splits,"clean_tolerance":.005,
            "certificate_parameters":{"eps":.01,"eta":.1,"retention":.9}}
