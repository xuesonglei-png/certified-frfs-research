"""Check final local artifact relationships without rerunning experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return (ROOT / name).read_text(encoding='utf-8')

def main():
    tex = read('paper-round3/main.tex') + read('paper-round3/results.tex')
    bib = read('paper-round3/references.bib')
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    citations = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', tex)
                 for key in group.split(',')}
    assert keys == citations and len(keys) == 16, (keys - citations, citations - keys)
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', tex))
    assert refs <= labels, refs-labels
    reader = PdfReader(ROOT / 'paper-round3/main.pdf')
    text = '\n'.join(page.extract_text() for page in reader.pages)
    for required in ('Songlei Xue', 'Zhongying Suo', 'suozhongying@sohu.com',
                     'zaiyujian0101@163.com', 'This research received no external funding.'):
        assert required in text, required
    assert 'Anonymous research manuscript' not in text
    log = read('paper-round3/main.log')
    assert not any(marker in log for marker in ['Overfull', 'undefined references',
                                               'Citation `', 'LaTeX Error'])
    tests = json.loads(read('results/round3/test-run-final.json'))
    assert tests['exit_code'] == 0
    for name, expected in tests['source_and_test_hashes'].items():
        assert digest(ROOT / name) == expected, name
    assert '303 passed' in read('results/round3/test-run-final.log')
    highlights = [line[2:] for line in read('submission/round3/highlights.txt').splitlines()
                  if line.startswith('- ')]
    assert 3 <= len(highlights) <= 5 and all(len(line) <= 85 for line in highlights)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(ROOT / 'submission/round3/highlights.docx') as doc:
        tree = ElementTree.fromstring(doc.read('word/document.xml'))
        word_text = ''.join(node.text or '' for node in tree.findall('.//w:t', ns))
    assert all(line in word_text for line in highlights)
    files = ['paper-round3/main.tex', 'paper-round3/main.pdf', 'paper-round3/main.bbl',
             'paper-round3/references.bib', 'paper-round3/results.tex',
             'paper-round3/figures/gap-validation.png',
             'submission/round3/highlights.docx', 'submission/round3/cover-letter-draft.docx']
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'pdf_pages': len(reader.pages), 'references': len(keys),
              'missing_citations': [], 'unused_references': [], 'undefined_labels': [],
              'source_and_tests_match_retained_303_pass_run': True,
              'highlights_character_counts': [len(line) for line in highlights],
              'artifact_sha256': {name: digest(ROOT/name) for name in files},
              'nonblocking_notes': [
                  'One underfull TeX paragraph; inspected visually without clipping.',
                  'CCCG2008 entry has no verified page range; BibTeX emits empty-pages warning.',
                  'Skill bibliography checker exits 0 with DOI recommendations for sources having verified URLs.',
                  'No new experiment or test-suite run performed by this check.'],
              'visual_review_record': 'research/round3/final-artifact-qa.md'}
    (ROOT/'results/round3/artifact-integration.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status','pdf_pages','references',
                       'source_and_tests_match_retained_303_pass_run','highlights_character_counts']}))

if __name__ == '__main__':
    main()
