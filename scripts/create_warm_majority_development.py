"""Version an ALL15 equal-budget warm-start extension, queued after cold review."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(path,text):
    p=ROOT/path;assert not p.exists(),p;p.write_text(text,encoding='utf-8')


def main():
    runner=(ROOT/'scripts/run_direct_majority_frs_development.py').read_text(encoding='utf-8')
    runner=runner.replace('from fit_fixed_face_majority_milp import fit_majority_face_milp',
        'from fit_warm_majority_face_milp import fit_warm_majority_face_milp')
    runner=runner.replace('fit_majority_face_milp(', 'fit_warm_majority_face_milp(')
    runner=runner.replace('direct-majority-frs-development','warm-majority-frs-development')
    write('scripts/run_warm_majority_frs_development.py',runner)
    audit=(ROOT/'scripts/audit_direct_majority_frs_development.py').read_text(encoding='utf-8')
    audit=audit.replace('from run_direct_majority_frs_development import','from run_warm_majority_frs_development import')
    audit=audit.replace('from audit_fixed_face_majority_milp import check_fit','from audit_warm_majority_mip import check_fit')
    write('scripts/audit_warm_majority_frs_development.py',audit)
    prep=(ROOT/'scripts/prepare_direct_majority_frs_development.py').read_text(encoding='utf-8')
    prep=prep.replace('from run_direct_majority_frs_development import','from run_warm_majority_frs_development import')
    prep=prep.replace('direct-majority-frs-development','warm-majority-frs-development')
    prep=prep.replace('direct-majority-FRS-development','warm-majority-FRS-development')
    prep=prep.replace('run_direct_majority_frs_development.py','run_warm_majority_frs_development.py')
    prep=prep.replace('audit_direct_majority_frs_development.py','audit_warm_majority_frs_development.py')
    prep=prep.replace("tests['tests_passed']==2","tests['tests_passed']==1")
    needle="    parents=prerequisites();status('freezing')"
    assert prep.count(needle)==1
    prep=prep.replace(needle,"""    review_path=ROOT/'research/direct-majority-FRS-development-review.json'
    status('waiting_for_full_cold_majority_review')
    while not review_path.exists():
        previous=ROOT/'results/direct-majority-frs-development-chain-status.json'
        if previous.exists() and json.loads(previous.read_text())['phase']=='failed':
            raise RuntimeError('Cold majority chain failed; preserve failure before warm extension')
        time.sleep(10)
    review=json.loads(review_path.read_text())
    cold=ROOT/'results/direct-majority-frs-development'
    assert len(review['all_dataset_rows'])==15 and review['audit_sha256']==digest(cold/'complete-audit.json')
    assert json.loads((cold/'complete-audit.json').read_text())['summary_sha256']==digest(cold/'complete.json')
    parents=prerequisites();status('freezing')""")
    needle="    paths += [TESTS,ROOT/'research/warm-majority-FRS-development-protocol.md']"
    assert prep.count(needle)==1
    prep=prep.replace(needle,needle+"\n    paths += [review_path,cold/'complete.json',cold/'complete-audit.json']")
    write('scripts/prepare_warm_majority_frs_development.py',prep)
    protocol=(ROOT/'research/direct-majority-FRS-development-protocol.md').read_text(encoding='utf-8')
    protocol+='''

## Equal-budget known feasible starting assignment extension

This version supplies original feasible seed weights, actual qualified query
indicators, the three closest within-class friends and two closest enemy
exceptions BEFORE running Highs1.15.1. Free certifying epigraph variables are
completed by the solver. The nominal objective, original FRS geometry,
radius/tolerance/retention, starting actual weight point, 120-second budget,
relative gap .01 and all15 schedule stay the same for both matched methods.
Both warm arms use single-thread Highs with fixed random_seed0. No test labels
or extra restarts. This is a solver initialization extension, not a new theory.
Original scipy cold-MIP arms are retained as independently audited full-cohort
evidence. No claim that the earlier scipy solver received our feasible seed.

The warm chain waits for the COMPLETE audited cold ALL15 review, preserving
every result before proceeding. It freezes its own code after that prerequisite
but this protocol and tests are committed prospectively before warm fits.
No gating, dataset selection or tuning based on those preceding scores.
Compare entire corresponding cohorts after both full audits, including
non-descent fallbacks and all negative rows. Significance still requires new
independent data after complete method/tuning freeze.
'''
    write('research/warm-majority-FRS-development-protocol.md',protocol)


if __name__=='__main__':main()
