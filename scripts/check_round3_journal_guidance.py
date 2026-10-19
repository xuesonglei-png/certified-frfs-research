"""Retain official guidance access results; never infer content from a failure."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import html
import json
import re
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'submission' / 'round3' / 'journal-guidance'
URLS = {
    'fss-guide': 'https://www.sciencedirect.com/journal/fuzzy-sets-and-systems/publish/guide-for-authors',
    'fss-scope': 'https://www.sciencedirect.com/journal/fuzzy-sets-and-systems/about/aims-and-scope',
    'elsevier-ai': 'https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals',
    'elsevier-highlights': 'https://www.elsevier.com/researcher/author/tools-and-resources/highlights',
}

def fetch(item):
    key, url = item
    row = {'key': key, 'url': url,
           'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=30)
        row.update(status=response.status_code, final_url=response.url,
                   sha256=hashlib.sha256(response.content).hexdigest())
        if response.ok:
            (DEST / (key + '.html')).write_text(response.text, encoding='utf-8')
            body = re.sub(r'<(script|style|noscript)\b[^>]*>.*?</\1>', '', response.text, flags=re.S | re.I)
            plain = html.unescape(re.sub(r'<[^>]+>', '\n', body))
            plain = '\n'.join(line.strip() for line in plain.splitlines() if line.strip())
            (DEST / (key + '.txt')).write_text(plain, encoding='utf-8')
            row['text_characters'] = len(plain)
    except requests.RequestException as error:
        row['error'] = str(error)
    return row

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch, URLS.items()))
    (DEST / 'access.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(records, indent=2))
