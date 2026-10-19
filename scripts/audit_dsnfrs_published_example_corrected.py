"""Rational-distance/Decimal replay of every retrospective example check."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import numpy as np
from recheck_dsnfrs_published_example import ROOT,TABLE1,PRINTED,rational_profile


def main():
    path=ROOT/'research/dsnfrs-published-example-corrected-recheck.json'
    report=json.loads(path.read_text()); assert report['passed'] and report['table1']==TABLE1
    assert report['table3_own_class_printed']==PRINTED and len(report['rows'])==360
    for rel,sha in report['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    matched=[]; max_float_error=0.
    for row in report['rows']:
        exact=rational_profile(row['features_zero_based'],beta=Fraction(str(row['beta'])),
            ddof=row['ddof'],zero_lambda=Fraction(str(row['zero_variance_lambda'])))
        rounded=[r['rounded4'] for r in exact]
        assert row['rounded4']==rounded
        exact_values=np.array([float(r['lower_decimal80']) for r in exact])
        discrepancy=float(np.max(abs(np.array(row['actual'])-exact_values)))
        assert discrepancy<1e-13
        max_float_error=max(max_float_error,discrepancy)
        match=rounded==[format(float(v),'.4f') for v in PRINTED]
        assert row['matches_all_15_printed_own_class_values']==match
        if match: matched.append(row)
    assert len(matched)==report['matches']==12
    assert {tuple(r['features_zero_based']) for r in matched}=={(0,)}
    assert all(r['ddof']==1 for r in matched)
    result={'passed':True,'configurations':360,'exact_own_class_values':5400,
        'compatible_configurations':12,'max_float_vs_decimal_error':max_float_error,
        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'rational_formula_example_replay_not_author_setting_identification_or_prediction_comparison'}
    (ROOT/'research/dsnfrs-published-example-corrected-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
