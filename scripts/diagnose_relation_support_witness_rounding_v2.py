"""Record the first scalar/preweighted witness mismatch; no model changes."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from certified_frfs.datasets import load_dataset
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.validated_pool import preprocess_fit
from audit_relation_support_selector_v1 import independent_protected

threadpool_limits(1)
folder = ROOT/"results"/"relation-support-selector-v3"
saved = json.loads((folder/"haberman.json").read_text(encoding="utf-8"))
X, y = load_dataset("haberman")
for outer in saved["rows"]:
    raw, labels = X[outer["train_indices"]], y[outer["train_indices"]]
    for family in outer["fit"]["inner"]:
        for inner in family["folds"]:
            A, _ = preprocess_fit(raw[inner["fit_indices"]], raw[inner["validation_indices"]])
            yy = labels[inner["fit_indices"]]
            w = np.ones(A.shape[1])/np.sqrt(A.shape[1])
            bounds = MajorityTerms(A, yy, weights=w, eps=.01).bounds(range(A.shape[1]), allowed_enemies=2)
            production_mask = bounds > 1e-10
            scalar_mask = independent_protected(A, yy, w)
            differing = np.flatnonzero(production_mask != scalar_mask)
            if not len(differing):
                continue
            preweighted = cdist(A*w, A*w, "cityblock")
            i = int(differing[0])
            friends = np.flatnonzero((yy == yy[i]) & (np.arange(len(yy)) != i))
            direct = np.array([float(np.sum(abs(A[j]-A[i])*w)) for j in friends])
            order_before = friends[np.lexsort((friends, preweighted[i,friends]))][:3]
            order_after = friends[np.lexsort((friends, direct))][:3]
            ids = sorted(set(order_before.tolist()+order_after.tolist()))
            report = {"dataset": "haberman", "outer_fold": outer["fold"], "inner_fold": inner["fold"], "object_relative_to_inner_train": i,
                      "production_protected": bool(production_mask[i]), "original_scalar_audit_protected": bool(scalar_mask[i]),
                      "production_padded_margin": float(bounds[i]), "preweighted_friend_order": order_before.tolist(),
                      "direct_friend_order": order_after.tolist(), "candidate_distances": [{"index": j, "scaled_before_difference": float(preweighted[i,j]), "difference_before_scaled": float(np.sum(abs(A[j]-A[i])*w))} for j in ids],
                      "cause": "binary_floating_point_order_of_operations_changes_fixed_witness_ties", "action": "preserve_model_artifacts_and_original_failure_use_separate_independent_scale_before_subtract_replay"}
            (folder/"haberman-witness-rounding-diagnostic-v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            raise SystemExit(0)
raise RuntimeError("Expected witness mismatch not found")
