"""Replay updated support sources under both shared classifier objectives."""
import audit_balanced_robust_controls as original

original.SOURCE = original.ROOT/'results/nested-balanced-joint'
original.FOLDER = original.ROOT/'results/nested-balanced-joint-robust-controls'

if __name__ == '__main__':
    original.main()
