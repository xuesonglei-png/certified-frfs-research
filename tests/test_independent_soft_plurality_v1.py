import numpy as np
from independent_soft_plurality_v1 import objective
from soft_plurality_objective_v1 import terms_and_temperature,plurality_objective
from density_calibrated_soft5 import objective as own_mass
from test_dynamic_certificate_metric_v1 import table


def test_independent_root_multiclass_gradient_at_low_and_regular_temperature():
    A,y=table();w=np.array([.44,.56]);terms,tau,_=terms_and_temperature(A,y,w)
    for temperature in (tau,.0005,.05):
        actual,gradient=plurality_objective(w,terms,y,temperature)
        expected,independent_gradient=objective(A,y,w,temperature)
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(gradient,independent_gradient,atol=2e-7,rtol=2e-6)
        base,bgrad=own_mass(w,terms,temperature)
        independent,igrad=objective(A,y,w,temperature,family='own_mass')
        np.testing.assert_allclose(base,independent,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(bgrad,igrad,atol=2e-7,rtol=2e-6)
