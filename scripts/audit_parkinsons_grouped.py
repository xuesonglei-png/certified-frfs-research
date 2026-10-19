"""Reuse the independent audit with patient-disjoint outer split assertions."""
import audit_newdata_compact as audit
from run_parkinsons_grouped import configure, PatientOuterFolds, grouped_sources
import run_newdata_compact as experiment


if __name__=='__main__':
    configure()
    audit.FOLDER=experiment.FOLDER
    audit.SOURCES={'parkinsons':experiment.SOURCES['parkinsons']}
    audit.StratifiedKFold=PatientOuterFolds
    audit.sources=grouped_sources
    audit.ARCHIVE=experiment.ROOT/'output/parkinsons-grouped-evidence.zip'
    audit.main()
