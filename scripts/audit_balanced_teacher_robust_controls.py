"""Scalar margins/objectives/FRS/predictions against balanced teacher source."""
import audit_balanced_robust_controls as original

original.SOURCE = original.ROOT/'results/nested-balanced-teacher-backward'
original.FOLDER = original.ROOT/'results/balanced-teacher-robust-controls'

if __name__ == '__main__':
    original.main()
