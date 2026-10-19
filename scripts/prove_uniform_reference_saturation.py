"""Exact symbolic MGF and rational constants for a saturation lemma."""
import hashlib
import json
from pathlib import Path
import sympy as sp
from run_nested_geometry_forest import ROOT


def main():
    z,t,a = sp.symbols('z t a',positive=True)
    raw_mgf = sp.integrate(2*(1-z)*sp.exp(-t*z),(z,0,1))
    expected = 2*(t-1+sp.exp(-t))/t**2
    assert sp.simplify(raw_mgf-expected) == 0
    truncated = 2*a-a**2+sp.integrate(2*(1-z)*sp.exp(-t*(z-a)),(z,a,1))
    expected_truncated = 2*a-a**2+2*((1-a)*t-1+sp.exp(-t*(1-a)))/t**2
    assert sp.simplify(truncated-expected_truncated) == 0
    assert sp.simplify(sp.integrate(2*(1-z)*(z-a),(z,a,1))-(1-a)**3/3) == 0
    # exp(6)>100 and exp(147/25)>100 by finite positive series terms.
    nominal_series = sum(sp.Rational(6)**k/sp.factorial(k) for k in range(6))
    radius_series = sum(sp.Rational(147,25)**k/sp.factorial(k) for k in range(6))
    assert nominal_series > 100 and radius_series > 100
    assert sp.Rational(31,4)**2 > 60 and sp.Rational(6)*sp.Rational(31,4) < 47
    # e<3 follows from k!>=2^(k-1), with strict inequality for some k.
    # Thus exp(6sqrt60)<3^47. No floating-point comparison is used.
    nominal_mgf_upper = sp.Rational(501,1800)
    robust_mgf_upper = sp.Rational(99,2500)+sp.Rational(489,1800)
    nominal_failure = sp.Rational(200*199,2)*3**47*nominal_mgf_upper**60
    robust_failure = sp.Rational(200*199,2)*3**47*robust_mgf_upper**60
    assert nominal_failure < sp.Rational(1,10**6)
    assert robust_failure < sp.Rational(1,1000)
    payload = {'verified_symbolic':True,'raw_MGF':str(expected),'truncated_MGF':str(expected_truncated),
        'truncated_mean':str((1-a)**3/3),'n':200,'d':60,'t':6,'eps':'.01',
        'nominal_failure_rational_upper':str(nominal_failure),'robust_failure_rational_upper':str(robust_failure),
        'nominal_failure_display':str(nominal_failure.evalf(18)),'robust_failure_display':str(robust_failure.evalf(18)),
        'strict_exact_nominal_upper':'1/1000000','strict_exact_robust_upper':'1/1000',
        'scope':'independent_uniform_coordinates_example_all_labelings_full_reference_dependency_saturation_even_under_shared_box',
        'limits':'Specific_iid_uniform_DGP_not_UCI_data_model; not_novelty_claim_or_new_classifier_advantage',
        'sympy_version':sp.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'results/uniform-reference-saturation-lemma.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print({k:v for k,v in payload.items() if not k.endswith('rational_upper')},flush=True)


if __name__ == '__main__': main()
