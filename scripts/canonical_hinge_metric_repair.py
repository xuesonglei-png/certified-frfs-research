"""Repair and preserve ACTUAL metric coordinates, without threshold changes."""
import numpy as np


def repair_actual_metric(C,rhs,actual,anchor):
    actual=np.array(actual,float,copy=True);anchor=np.asarray(anchor,float);trace=[];total_blend=0.
    if not len(C):return actual,trace,total_blend
    assert np.all(C@anchor>rhs),'Strict feasible anchor is required'
    for attempt in range(8):
        slack=C@actual-rhs
        if np.all(slack>=0):return actual,trace,total_blend
        anchor_slack=C@anchor-rhs;bad=slack<0
        denominator=anchor_slack[bad]-slack[bad]
        fraction=float(min(1.,max(10.**(attempt-12),1.01*np.max(-slack[bad]/denominator))))
        before=float(slack.min());actual=(1-fraction)*actual+fraction*anchor
        total_blend=1-(1-total_blend)*(1-fraction)
        trace.append({'attempt':attempt,'anchor_fraction':fraction,'min_slack_before':before,
                      'min_slack_after':float((C@actual-rhs).min())})
    if not np.all(C@actual>=rhs):
        actual=anchor.copy();total_blend=1.;trace.append({'attempt':'explicit_anchor','anchor_fraction':1.,
            'min_slack_after':float((C@actual-rhs).min())})
    assert np.all(C@actual>=rhs)
    return actual,trace,total_blend
