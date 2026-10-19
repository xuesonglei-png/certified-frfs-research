"""Replay exact-box deployed certificates and predictions independently."""
import audit_robust_linear_refit as audit

if __name__ == '__main__':
    audit.FOLDER = audit.ROOT / 'results/exact-box-linear-refit'
    audit.main()
