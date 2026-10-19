"""New status destination only; frozen signed experiment is unchanged."""
import json
import sys
import traceback
import continue_signed_classwise_admission as chain

chain.STATUS = chain.ROOT/'results/signed-classwise-admission-chain-v2-status.json'


if __name__ == '__main__':
    original = chain.ROOT/'results/signed-classwise-admission-chain-status.json'
    previous = json.loads(original.read_text()); assert previous['phase'] == 'failed'
    note = {'original_failed_status': previous, 'wrapper_sha256': chain.digest(__file__),
            'original_chain_sha256': chain.digest(chain.ROOT/'scripts/continue_signed_classwise_admission.py'),
            'scope': 'Resume unchanged experiment after JSON-only upstream audit repair; old failed status retained.'}
    (chain.ROOT/'research/signed-classwise-chain-resume-provenance.json').write_text(json.dumps(note, indent=2), encoding='utf-8')
    try:
        chain.main(int(sys.argv[1]))
    except Exception as error:
        chain.status('failed', error=str(error)); traceback.print_exc(); raise
