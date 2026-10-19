import json
import audit_realized_budget_diagnostic as audit
from audit_common_fuzzy_classifier import independent_fit,score

if __name__=='__main__':
    audit.FOLDER=audit.ROOT/'results/realized-fuzzy-budget'
    # Audit requires common raw path records, which are unchanged across classifier runs.
    audit.independent_fit=independent_fit;audit.score=score;audit.main()
