import numpy as np
import certified_frfs.selective_support_selector_v3 as selector


def test_recipe_survives_fold_local_rank_changes(monkeypatch):
    """Regression: fit procedures must survive feature-ID changes across folds."""
    X=np.random.default_rng(21).random((30,6)); y=np.repeat([0,1],15)
    monkeypatch.setattr(selector,"relations",lambda A,y:{f:np.ones(6)/np.sqrt(6) for f in selector.FAMILIES})
    monkeypatch.setattr(selector,"_ranking",lambda A,y,w,f,seed:np.roll(np.arange(6),seed%6).tolist())
    def candidate(A,y,w,ranking,count):
        fs=sorted(ranking[:count])
        return {"features":fs,"weights":(np.ones(count)*np.sqrt(6)/count).tolist(),"joint_feasible":True}
    monkeypatch.setattr(selector,"_candidate",candidate)
    monkeypatch.setattr(selector,"query_scores",lambda A,y,B,truth,c:{"clean_BA":.8,"query_certified_BA":.7,"query_coverage":.75})
    fit=selector.fit_outer(X,y,seed=9,inner_splits=3)
    chosen=fit["chosen"]; assert chosen["recipe"]["count"]==3
    supports=[b["families"][chosen["family"]]["candidates"][chosen["recipe_index"]]["candidate"]["features"] for b in fit["inner"]]
    assert len({tuple(fs) for fs in supports})==3
    assert fit["selected"]["features"]==selector.support(fit["refit_ranking"],chosen["recipe"])


def test_rejected_outer_refit_is_kept_separate_from_fallback(monkeypatch):
    X=np.random.default_rng(21).random((30,6)); y=np.repeat([0,1],15)
    monkeypatch.setattr(selector,"relations",lambda A,y:{f:np.ones(6)/np.sqrt(6) for f in selector.FAMILIES})
    monkeypatch.setattr(selector,"_ranking",lambda *args:list(range(6)))
    def candidate(A,y,w,ranking,count):
        return {"features":sorted(ranking[:count]),"weights":(np.ones(count)*np.sqrt(6)/count).tolist(),
                "joint_feasible":len(A)<30 or count==6}
    monkeypatch.setattr(selector,"_candidate",candidate)
    monkeypatch.setattr(selector,"query_scores",lambda A,y,B,truth,c:{"clean_BA":.8,"query_certified_BA":.7,"query_coverage":.75})
    fit=selector.fit_outer(X,y,seed=9,inner_splits=3)
    assert not fit["refit_attempt"]["joint_feasible"]
    assert len(fit["refit_attempt"]["features"])==3
    assert fit["selected"]["outer_refit_fallback"]
    assert fit["selected"]["requested_count"]==3 and fit["selected"]["realized_count"]==6
