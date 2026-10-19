"""Create a private review bundle and an auditable local provenance record."""
from pathlib import Path
import hashlib
import json
import platform
import zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    evidence={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'python':platform.python_version(),
        'purpose':'private local review bundle; no public submission or upload',
        'checks':json.loads((ROOT/'results/reproduction-audit.json').read_text()),
        'data_manifest_sha256':sha(ROOT/'data/manifest.json'),
        'requirements_sha256':sha(ROOT/'requirements-lock.txt'),
        'manuscript_pdf_sha256':sha(ROOT/'paper/main.pdf'),
        'datasets':{name:{'array_sha256':row['array_sha256'],'n':row['n_samples'],'d':row['n_features']}
                    for name,row in manifest.items()},
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
                        for directory in ['src','scripts','tests']
                        for p in sorted((ROOT/directory).rglob('*.py')) if '__pycache__' not in p.parts}
    }
    (ROOT/'results/provenance.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    output=ROOT/'output'
    output.mkdir(exist_ok=True)
    target=output/'certified-frfs-research-v1.zip'
    # Third-party paper PDFs and raw webpage dumps remain in the local library;
    # the review ZIP contains notes and citations, not redistributable copies.
    directories=['src','scripts','tests','paper','submission','data/processed','results']
    files=[ROOT/'README.md',ROOT/'研究结果与投稿准备.md',ROOT/'requirements-lock.txt',ROOT/'pyproject.toml',ROOT/'data/manifest.json',ROOT/'data/download_datasets.py']
    for directory in directories:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files.extend(p for p in (ROOT/'research').glob('*.md'))
    files.extend(p for p in (ROOT/'research/literature').glob('*.md'))
    files.append(ROOT/'research/literature/library.json')
    banned_suffixes={'.pyc','.aux','.bbl','.blg','.log','.out','.gz','.spl','.tmp'}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):
            relative=path.relative_to(ROOT)
            if '__pycache__' in relative.parts or 'archive' in relative.parts or path.suffix in banned_suffixes or '.partial.' in path.name:
                continue
            z.write(path,str(relative))
    report={'bundle':str(target.relative_to(ROOT)),'sha256':sha(target),
            'bytes':target.stat().st_size,'entries':len(zipfile.ZipFile(target).namelist())}
    (output/'bundle-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
