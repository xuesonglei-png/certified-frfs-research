"""Independent nested certificates, margins, teachers and prediction replay."""
import audit_nested_triangular_repair as repair

repair.original.FOLDER = repair.original.ROOT / 'results/nested-fuzzy-logistic'

if __name__ == '__main__':
    repair.original.main()
