"""Smooth concave support penalty on the original normalized simplex."""
import numpy as np


def penalty(point, delta=.01):
    point = np.asarray(point, float)
    if not len(point) or not np.isfinite(point).all() or point.min() < 0 or not np.isfinite(delta) or delta <= 0:
        raise ValueError('finite nonnegative coordinates and positive scale required')
    return float(np.log1p(point/delta).mean()/np.log1p(1/delta))


def tangent(point, delta=.01):
    point = np.asarray(point, float); value = penalty(point, delta)
    gradient = 1/((delta+point)*len(point)*np.log1p(1/delta))
    intercept = float(value-gradient@point)
    return gradient, intercept
