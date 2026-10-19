"""Public metadata/code discovery only; no author messaging or private access."""
import json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'research/literature/public-code-discovery-20261010'
URLS = {
    'github-mafrfs':'https://api.github.com/search/repositories?q=MAFRFS',
    'github-margin-fuzzy':'https://api.github.com/search/repositories?q=%22margin-aware%22+fuzzy',
    'github-rfrr':'https://api.github.com/search/repositories?q=RFRR+fuzzy',
    'github-suping-xu':'https://api.github.com/search/repositories?q=fuzzy+user%3ASupingXu',
    'ieee-supplement-rest':'https://ieeexplore.ieee.org/rest/document/11218698/supplemental-media',
    'ieee-media-page':'https://ieeexplore.ieee.org/document/11218698/media',
    'arxiv-mafrfs-html':'https://arxiv.org/html/2505.15250v1',
}

def fetch(item):
    name,url = item; start = time.perf_counter()
    try:
        response = requests.get(url,timeout=25,headers={'User-Agent':'Academic-reproducibility-evidence/1.0','Accept':'application/json' if 'api.github.com' in url else '*/*'})
        content = response.content; suffix = '.json' if 'json' in response.headers.get('Content-Type','') else '.html'
        output = FOLDER/(name+suffix); output.write_bytes(content)
        result = {'name':name,'url':url,'final_url':response.url,'status':response.status_code,'bytes':len(content),
            'content_type':response.headers.get('Content-Type'),'sha256':hashlib.sha256(content).hexdigest(),
            'artifact':output.relative_to(ROOT).as_posix(),'seconds':time.perf_counter()-start}
        if 'api.github.com' in url:
            data = response.json(); result['total_count'] = data.get('total_count')
            result['repositories'] = [{'full_name':r['full_name'],'url':r['html_url'],'description':r.get('description'),
                'default_branch':r['default_branch']} for r in data.get('items',[])]
            if response.status_code != 200: result['message'] = data.get('message')
    except Exception as error:
        result = {'name':name,'url':url,'error':type(error).__name__+': '+str(error),'seconds':time.perf_counter()-start}
    print(result,flush=True); return result

if __name__ == '__main__':
    FOLDER.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [f.result() for f in as_completed([pool.submit(fetch,item) for item in URLS.items()])]
    result = {'date':'2026-10-10','rows':rows,'scope':'public_artifact_discovery_not_verified_baseline_fidelity; failed_access_not_nonexistence'}
    (FOLDER/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
