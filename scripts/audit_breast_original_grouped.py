import audit_newdata_compact as audit
from run_breast_original_grouped import configure,IdentifierFolds,grouped_sources
import run_newdata_compact as experiment


if __name__=='__main__':
    configure();audit.FOLDER=experiment.FOLDER
    audit.SOURCES={'breast_original':experiment.SOURCES['breast_original']}
    audit.StratifiedKFold=IdentifierFolds;audit.sources=grouped_sources
    audit.ARCHIVE=experiment.ROOT/'output/breast-original-grouped-evidence.zip'
    audit.main()
