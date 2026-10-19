import audit_robust_linear_refit as audit

if __name__ == '__main__':
    audit.FOLDER = audit.ROOT / 'results/distilled-box-linear-refit'
    audit.main()
