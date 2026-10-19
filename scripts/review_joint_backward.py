"""Dataset-paired accuracy/compression changes; no selected-outcome headline."""
import json
from review_box_iterations import ROOT, paired


def main():
    folder = ROOT/'results/nested-joint-backward'
    latest = json.loads((folder/'complete-review.json').read_text())
    previous = json.loads((folder.parent/'nested-balanced-joint/complete-review.json').read_text())
    old = {r['dataset']: r for r in previous['rows']}
    accuracy = [r['clean']-old[r['dataset']]['clean'] for r in latest['rows']]
    compression = [r['removed']-old[r['dataset']]['removed'] for r in latest['rows']]
    report = {'vs_balanced_joint_accuracy': paired(accuracy),
              'vs_balanced_joint_compression': paired(compression),
              'deployment_review': latest,
              'limitations': 'Repeatedly exposed exploratory comparisons; compression gain alone does not establish task superiority.'}
    (folder/'paired-mechanism-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'deployment_review'}, flush=True)


if __name__ == '__main__':
    main()
