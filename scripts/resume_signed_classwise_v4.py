"""Correct Path metadata in resume wrapper; numerical experiment unchanged."""
import json
import sys
import traceback
from pathlib import Path
import continue_signed_classwise_admission as chain

chain.STATUS = chain.ROOT/'results/signed-classwise-admission-chain-v4-status.json'


if __name__ == '__main__':
    previous = json.loads((chain.ROOT/'results/signed-classwise-admission-chain-status.json').read_text())
    assert previous['phase'] == 'failed'
    note = {'original_failed_status': previous, 'wrapper_sha256': chain.digest(Path(__file__)),
            'failed_wrapper_sha256': chain.digest(chain.ROOT/'scripts/resume_signed_classwise_after_star_v3.py'),
            'failed_wrapper_log_sha256': chain.digest(chain.ROOT/'results/signed-classwise-v2-launch-error.log'),
            'original_chain_sha256': chain.digest(chain.ROOT/'scripts/continue_signed_classwise_admission.py'),
            'scope': 'Resume unchanged experiment; repair str-vs-Path in execution metadata only; all prior failures retained.'}
    (chain.ROOT/'research/signed-classwise-chain-resume-v4-provenance.json').write_text(json.dumps(note, indent=2), encoding='utf-8')
    try:
        chain.main(int(sys.argv[1]))
    except Exception as error:
        chain.status('failed', error=str(error)); traceback.print_exc(); raise
