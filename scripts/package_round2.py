"""Build a separate local second-cycle review bundle without replacing v1."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = {ROOT/p for p in ['README.md', '研究结果与投稿准备.md', 'pyproject.toml',
        'requirements-lock.txt', 'data/manifest.json', 'data/download_datasets.py',
        'research/PROGRESS.md', 'research/data-and-baselines.md',
        'results/real-subtable-verification.json']}
    for directory, pattern in [('src', '*.py'), ('tests', '*.py'), ('scripts', '*.py'),
                               ('research/round2', '*'), ('results/round2', '*'),
                               ('data/processed', '*.npz')]:
        files.update(p for p in (ROOT/directory).rglob(pattern) if p.is_file())
    files.update((ROOT/'results/primary').glob('*-s17-f0.json'))
    files.update((ROOT/'research/literature').glob('*.md'))
    files.add(ROOT/'research/literature/library.json')
    files = sorted(p for p in files if '__pycache__' not in p.parts)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    output = ROOT/'output'
    output.mkdir(exist_ok=True)
    target = output/'certified-frfs-round2-exploration.zip'
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Private local second-cycle review and reproduction; no public upload or submission',
        first_cycle_snapshot='Separate: certified-frfs-research-v1.zip is not modified',
        files_sha256=hashes,
        boundary='Includes original first folds and induced small tables as inputs; does not replace 132-split primary experiment or manuscript')
    instructions = '''# 第二轮独立研究复现包

先阅读 research/round2/研究进展与决策.md。
本包包含第二轮理论、代码、测试、原始输出，以及复用的15个首折与小子表输入。
第一轮完整132分折主实验、英文内稿和旧ZIP是独立快照，不在本包重新发布。
研究性比较采用不同模型时须分开解释；浮点验证不等于形式化认证。
数据来源与许可见 data/manifest.json；不含第三方论文PDF。
results/round2/archive/ 为已排除的失败尝试，不能纳入有效结果。

安装：python -m venv .venv；再用 .venv 中的 python -m pip install -r requirements-lock.txt。
检验：.venv/Scripts/python.exe -m pytest -q。
各脚本入口见研究报告，运行前归档同名输出。
verify_round2_records.py是完整工作区的整合审计，还需要另行保存的第一轮ZIP及其output/bundle-manifest.json；本包保留其已执行记录integration-audit.json。
原始文件SHA-256见 ROUND2-MANIFEST.json。
'''
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('ROUND2-MANIFEST.json', json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
        archive.writestr('ROUND2-README.md', instructions)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    report = dict(bundle=target.relative_to(ROOT).as_posix(), sha256=sha(target),
                  bytes=target.stat().st_size, files=len(files)+2, contents_verified=True)
    (output/'round2-bundle-manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
