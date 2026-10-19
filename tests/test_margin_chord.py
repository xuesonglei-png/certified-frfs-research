"""Chord witnesses, Delta bounds, and separation of single/all-enemy scope."""
from types import SimpleNamespace

import numpy as np
import pytest

import certified_frfs.margin_chord as chord
from certified_frfs.margin_verification import exact_negative_loo_margin_milp


@pytest.mark.parametrize("case,expected,mixture", [
    ("strict_gain", .25, .25), ("shared_lambda", -.2, -.2), ("crossing", .1, -.15)])
def test_known_constructions_return_legal_brackets(case, expected, mixture):
    boxes = {
        "strict_gain": ([[0,0],[0,0],[1,0],[0,.75]], [[1,0],[0,0],[1,0],[1,.75]]),
        "shared_lambda": ([[0,0],[0,1],[1,0],[.4,.4]], [[0,0],[0,1],[1,0],[.4,.4]]),
        "crossing": ([[0,0],[0,0],[.5,0],[0,.6]], [[1,0],[.5,0],[1,0],[1,.6]]),
    }
    lo, hi = map(np.asarray, boxes[case])
    result = chord.chord_enemy_interval(lo,hi,0,3,[1,2],[0,1],[1,1])
    assert result["chord_solver_success"]
    assert result["lower_bound"] == pytest.approx(mixture, abs=1e-8)
    assert result["lower_bound"] <= expected+1e-8
    assert expected <= result["upper_bound"]+1e-8
    assert np.all(result["worst_case"] >= lo) and np.all(result["worst_case"] <= hi)
    assert result["posterior_gap"] <= result["delta"]+1e-8
    if case != "crossing":
        assert result["midpoint_separation"]
        assert result["delta"] == 0
        assert result["upper_bound"] == pytest.approx(expected)


@pytest.mark.parametrize("m", [3,5,9])
def test_tight_family_delta_and_posterior_gap_without_claiming_exact_anchor(m):
    r, rho = 1/(4*m), .75
    centers = (2*np.arange(1,m+1)-1)/(2*m)
    lo = np.vstack([[0,0],np.column_stack([centers-r,np.zeros(m)]),[0,rho]])
    hi = np.vstack([[1,0],np.column_stack([centers+r,np.zeros(m)]),[1,rho]])
    result=chord.chord_enemy_interval(lo,hi,0,m+1,list(range(1,m+1)),[0,1],[1,1])
    true = rho-r-1/(2*m)
    expected_mixture = rho-r-.5
    assert result["lower_bound"] == pytest.approx(expected_mixture, abs=1e-8)
    assert result["delta"] == pytest.approx(.5)
    assert result["lower_bound"] <= true <= result["upper_bound"]+1e-8
    assert result["posterior_gap"] <= .5+1e-8
    assert not result["midpoint_separation"]
    exact=exact_negative_loo_margin_milp((lo+hi)/2,[0]*(m+1)+[1],0,
                weights=[1,1],eps=(hi-lo)/2,clip=False,time_limit=5)
    assert exact["optimal"]
    assert -exact["lower_bound"] == pytest.approx(true, abs=2e-6)


def test_pair_bound_and_all_enemy_witness_have_distinct_scopes():
    lo=hi=np.array([[0],[.1],[.8],[.05]])
    result=chord.chord_enemy_interval(lo,hi,0,2,[1],[0],[1])
    assert result["lower_bound"] == pytest.approx(.7)
    assert result["upper_bound"] == pytest.approx(.7)
    assert result["all_enemy_margin_at_witness"] == pytest.approx(-.05)
    assert result["all_enemy_indices"] == [2,3]


def test_chord_primal_objective_is_never_used_as_certificate(monkeypatch):
    # Only this module's chord solve is mocked; mixture has its own solver.
    monkeypatch.setattr(chord,"linprog",lambda *args,**kwargs: SimpleNamespace(
        x=None,fun=1e9,success=False,status=1,message="Interrupted"))
    result=chord.chord_enemy_interval([[0,0],[0,0],[1,0],[0,.75]],
             [[1,0],[0,0],[1,0],[1,.75]],0,3,[1,2],[0,1],[1,1])
    assert not result["chord_has_incumbent"]
    assert result["numerical_opt_chord"] == 1e9
    assert result["lower_bound"] == pytest.approx(.25,abs=1e-8)
    assert result["upper_bound"] == pytest.approx(.25)


def test_zero_width_and_zero_weight_coordinates():
    lo=np.array([[.2,.3],[0,.8],[.9,.1]])
    hi=np.array([[.2,.3],[.1,.9],[1,.2]])
    result=chord.chord_enemy_interval(lo,hi,0,2,[1],[0,1],[0,1])
    assert result["delta"] == 0
    assert result["midpoint_separation"]
    assert result["lower_bound"] == pytest.approx(.1-.6,abs=1e-8)
    assert result["upper_bound"] == pytest.approx(.1-.6)


@pytest.mark.parametrize("seed", range(8))
def test_random_chord_brackets_delta_and_actual_witness_against_milp(seed):
    rng=np.random.default_rng(11400+seed)
    lo=rng.uniform(-.5,.5,(5,2))
    hi=lo+rng.uniform(.01,.9,(5,2))
    weights=rng.uniform(.1,1.5,2)
    result=chord.chord_enemy_interval(lo,hi,0,4,[1,2,3],[0,1],weights)
    exact=exact_negative_loo_margin_milp((lo+hi)/2,[0,0,0,0,1],0,
              weights=weights,eps=(hi-lo)/2,clip=False,time_limit=5)
    assert exact["optimal"]
    assert result["lower_bound"] <= -exact["upper_bound"]+2e-6
    assert result["upper_bound"] >= -exact["lower_bound"]-2e-6
    assert result["posterior_gap"] <= result["delta"]+2e-6
    z=result["worst_case"]
    assert np.all(z>=lo) and np.all(z<=hi)
    distances=abs(z-z[0])@weights
    assert result["upper_bound"] == pytest.approx(distances[4]-distances[1:4].min())
    assert result["lower_bound"] <= result["upper_bound"]+1e-8
