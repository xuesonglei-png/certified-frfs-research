import audit_vote_aligned_refinement as audit
import diagnose_support_exchange as experiment
from certified_frfs.certified_support_exchange import refine_support_exchange

audit.runner = experiment.runner
audit.refine_vote_aligned = refine_support_exchange


if __name__ == '__main__':
    audit.main(objective_descent=False, support_exchange=True)
