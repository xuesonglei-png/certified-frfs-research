"""Read public preprint paragraphs/math/link text using only the standard library."""
from html.parser import HTMLParser
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'research/literature/public-code-discovery-20261010'

class Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.output = []; self.links = []; self.skip = 0
    def handle_starttag(self,tag,attrs):
        data = dict(attrs)
        if tag in ('script','style'): self.skip += 1
        if tag in ('p','div','section','h1','h2','h3','h4','table'): self.output.append('\n')
        if tag == 'a' and data.get('href'): self.links.append(data['href'])
    def handle_endtag(self,tag):
        if tag in ('script','style'): self.skip -= 1
        if tag in ('p','div','section','h1','h2','h3','h4','table'): self.output.append('\n')
    def handle_data(self,text):
        if not self.skip: self.output.append(text)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = Extractor(); parser.feed((FOLDER/'arxiv-mafrfs-html.html').read_text(encoding='utf-8'))
    text = '\n'.join(' '.join(line.split()) for line in ''.join(parser.output).splitlines() if line.strip())
    (FOLDER/'arxiv-mafrfs-html.txt').write_text(text,encoding='utf-8')
    keywords = ('gaussian','euclidean','min-max','negation','t-conorm','supplementary','code','similarity relation')
    paragraphs = [line for line in text.splitlines() if any(word in line.lower() for word in keywords)]
    links = sorted(set(link for link in parser.links if any(word in link.lower() for word in ('github','gitee','supplement','code'))))
    external = sorted(set(link for link in parser.links if link.startswith(('http://','https://'))))
    result = {'links':links,'all_external_links':external,'paragraphs':paragraphs,'scope':'public_preprint_v1_text_not_published_supplement_or_verified_operator_defaults'}
    (FOLDER/'arxiv-operator-code-extract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({'links':links,'matching_paragraphs':len(paragraphs)},flush=True)
    for paragraph in paragraphs: print(paragraph[:1600],flush=True)
