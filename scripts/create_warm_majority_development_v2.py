"""Retain stopped warm queue; version only rank-tie audit/recovery dependency."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(path,text):
    p=ROOT/path;assert not p.exists(),p;p.write_text(text,encoding='utf-8')


def main():
    runner=(ROOT/'scripts/run_warm_majority_frs_development.py').read_text(encoding='utf-8')
    runner=runner.replace('warm-majority-frs-development','warm-majority-frs-development-v2')
    write('scripts/run_warm_majority_frs_development_v2.py',runner)
    audit=(ROOT/'scripts/audit_warm_majority_frs_development.py').read_text(encoding='utf-8')
    audit=audit.replace('from run_warm_majority_frs_development import','from run_warm_majority_frs_development_v2 import',1)
    audit=audit.replace('from audit_warm_majority_mip import check_fit','from audit_warm_majority_mip_v2 import check_fit',1)
    write('scripts/audit_warm_majority_frs_development_v2.py',audit)
    prep=(ROOT/'scripts/prepare_warm_majority_frs_development.py').read_text(encoding='utf-8')
    prep=prep.replace('from run_warm_majority_frs_development import','from run_warm_majority_frs_development_v2 import',1)
    prep=prep.replace('warm-majority-frs-development','warm-majority-frs-development-v2')
    prep=prep.replace('warm-majority-FRS-development','warm-majority-FRS-development-v2')
    prep=prep.replace('run_warm_majority_frs_development.py','run_warm_majority_frs_development_v2.py')
    prep=prep.replace('audit_warm_majority_frs_development.py','audit_warm_majority_frs_development_v2.py')
    prep=prep.replace('direct-majority-frs-development-chain-status.json','direct-majority-rank-recovery-v2-chain-status.json')
    write('scripts/prepare_warm_majority_frs_development_v2.py',prep)
    test=(ROOT/'tests/test_warm_majority_mip.py').read_text(encoding='utf-8')
    test=test.replace('from audit_warm_majority_mip import check_fit','from audit_warm_majority_mip_v2 import check_fit',1)
    write('tests/test_warm_majority_mip_v2.py',test)
    protocol=(ROOT/'research/warm-majority-FRS-development-protocol.md').read_text(encoding='utf-8')
    protocol+='''

## V2 audit and prerequisite correction

The original warm queue stopped when cold Transfusion rank-identity audit
failed; no warm real fit or sourcefreeze occurred. All original failed state
and logs retained. The cold science code is unchanged; v2 independently
permits nearest-witness identity alternatives ONLY inside a machine-epsilon
distance envelope, while retaining actual qualification/objective/certificate
checks. Supplied warm-start witness identities are independently validated
by that same check before their binary assignments are replayed. Actual
prediction and noise replay remain exact in their original tolerance.

The new warm-v2 queue waits for the complete audited ALL15 cold review via
direct-majority-rank-recovery-v2-chain-status.json. It must not interpret the
retained original cold failed status as the current recovery status. Its own
algorithm, known-feasible seed,120s time budget and all15 schedule are unchanged.
This revision is an audit correction, not performance tuning or tolerance
relaxation on classification, FRS loss, radius or majority preservation.
'''
    write('research/warm-majority-FRS-development-v2-protocol.md',protocol)


if __name__=='__main__':main()
