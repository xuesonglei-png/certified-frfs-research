"""Reproducible independent endpoint audit for the frozen V5 screen.

The audit logic is shared with the V4 review, but the model builder is
explicitly rebound to V5 before any embedding is constructed.  No solver is
called by the auditor; saved endpoints and old attack tables are replayed
against the actual shared-table constraints.
"""
from __future__ import annotations

import audit_shared_graph_v4_both_targets_independent as _v4
import independent_positive_graph_embedding_v4 as _embedding
import shared_graph_dependency_v5 as _v5


def main(only=None):
    _embedding.build = _v5.build
    _v4.FOLDER = _v4.ROOT / 'results/shared-graph-dependency-v5'
    _v4.__file__ = __file__
    return _v4.main(only)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=_v4.NAMES)
    args = parser.parse_args()
    main(args.only)
