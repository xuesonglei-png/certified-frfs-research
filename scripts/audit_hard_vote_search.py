import audit_vote_aligned_refinement as audit
import diagnose_hard_vote_search as experiment
from certified_frfs.hard_vote_search import refine_hard_vote

audit.runner = experiment.runner
audit.refine_vote_aligned = refine_hard_vote


if __name__ == '__main__':
    audit.main(objective_descent=False)
