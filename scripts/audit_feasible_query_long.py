"""Run the full fixed-certificate audit on the ten-second generation."""
import audit_feasible_query as audit
import diagnose_feasible_query as base
from diagnose_feasible_query_long import configure


if __name__ == '__main__':
    configure()
    audit.FOLDER = base.FOLDER
    audit.ARCHIVE_NAME = 'feasible-query-long-generation-evidence.zip'
    audit.main()
