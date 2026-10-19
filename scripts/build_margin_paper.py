"""Generate manuscript tables and bibliography from retained evidence."""
from pathlib import Path
import json,re,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'paper-round3'

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def escape(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')

def main():
    search=ROOT/'research/round3/search'
    metadata={}
    for file in ['crossref-classics-and-recent.json','crossref-initial.json']:
        for searchrow in json.loads((search/file).read_text(encoding='utf-8')):
            for item in searchrow.get('items',[]):
                if 'DOI' in item:metadata[item['DOI'].lower()]=item
    for row in json.loads((search/'verified-recent-records.json').read_text(encoding='utf-8')):
        if row.get('status')==200 and 'message' in row.get('result',{}):
            metadata[row['doi'].lower()]=row['result']['message']
    for key in ['Sion1958','Cook1971']:
        item=json.loads((search/(key+'.crossref.json')).read_text(encoding='utf-8'))['message']
        metadata[item['DOI'].lower()]=item
    item=json.loads((search/'margin2004_metadata.json').read_text(encoding='utf-8'))['message']
    metadata[item['DOI'].lower()]=item
    # Complete the abbreviated search record from the downloaded published
    # article's first page (search/knn_abstract2024.txt).
    kis=metadata['10.1007/s10115-024-02108-4']
    kis.update({'container-title':['Knowledge and Information Systems'],
                'volume':'66','page':'4825-4859','type':'journal-article'})
    kis['author'][0]['given']='Nicolò'
    needed={'GiladBachrach2004':'10.1145/1015330.1015352','Sion1958':'10.2140/pjm.1958.8.171',
            'Cook1971':'10.1145/800157.805047','Ranzato2025':'10.1145/3719027.3765140',
            'Fassina2024':'10.1007/s10115-024-02108-4'}
    bib=[]
    old=(ROOT/'paper/references.bib').read_text(encoding='utf-8')
    for key in ['Xu2026MarginAware','Wang2016Notes','Deer2015Comprehensive',
                'Zhao2013RFRR','Theerens2023Quantification']:
        match=re.search(r'@\w+\{'+key+r',.*?\n\}',old,re.S);assert match,key
        bib.append(match.group())
    for key,doi in needed.items():
        item=metadata[doi]
        authors=' and '.join(a['family']+', '+a.get('given','') for a in item['author'])
        kind='inproceedings' if item.get('type')=='proceedings-article' else 'article'
        venue='booktitle' if kind=='inproceedings' else 'journal'
        year=item.get('published',item.get('issued'))['date-parts'][0][0]
        fields={'author':authors,'title':item['title'][0],venue:item['container-title'][0],
                'year':year,'doi':doi,'url':'https://doi.org/'+doi}
        if item.get('page'):fields['pages']=item['page'].replace('-', '--')
        if item.get('volume'):fields['volume']=item['volume']
        bib.append('@'+kind+'{'+key+',\n'+',\n'.join('  '+k+' = {'+escape(v)+'}' for k,v in fields.items())+'\n}')
    if (search/'Voracek2022.verified.bib').exists():
        raw=(search/'Voracek2022.verified.bib').read_text(encoding='utf-8')
    else:
        source=(search/'pmlr_prototype.html').read_text(encoding='utf-8')
        raw=html.unescape(re.search(r'@InProceedings\{.*?</code>',source,re.S).group().removesuffix('</code>'))
        # Official PMLR BibTeX includes an unneeded long abstract.
        raw=raw[:raw.find('  abstract =')].rstrip(',\n ')+ '\n}'
        raw=raw.replace('pmlr-v162-voracek22a','Voracek2022')
    bib.append(raw)
    bib.append('@article{Weinberger2009,\n author={Weinberger, Kilian Q. and Saul, Lawrence K.},\n title={Distance Metric Learning for Large Margin Nearest Neighbor Classification},\n journal={Journal of Machine Learning Research},\n year={2009},\n volume={10},\n pages={207--244},\n url={https://www.jmlr.org/papers/v10/weinberger09a.html}\n}')
    # Curated additional entries are backed by the targeted full-text audit.
    extra=search/'theorem-additional-references.bib'
    if extra.exists():
        bib.append(extra.read_text(encoding='utf-8'))
    # A DOI already supplies a resolvable article link; avoid printing it twice.
    bib=[re.sub(r',\s*url\s*=\s*\{[^\n]+\}(?=\s*[,\n])','',entry)
         if re.search(r'\bdoi\s*=',entry) else entry for entry in bib]
    (DEST/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
    pilot=read('results/round3/margin-pilot.json')
    proto=read('results/round3/margin-prototypes.json')['summary']
    bench=read('results/round3/margin-mixture-benchmark.json')
    sat=read('results/round3/sat-reduction-audit.json')['summary']
    assert pilot['run_status']==bench['run_status']=='complete'
    gap=read('results/round3/margin-gap-benchmark.json')
    plot=gap['plot_data'];fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for name,label,marker in [('simple','Fixed vertical coordinates','o'),('nondegenerate','Positive-width boxes','x')]:
        points=[r for r in plot['tight_family'] if r['family']==name]
        if not points:
            if name=='nondegenerate':points=[r for r in plot['tight_family'] if r['family']!='simple']
        axes[0].plot([r['m'] for r in points],[r['gap_ratio'] for r in points],marker=marker,label=label,lw=1.2)
    axes[0].axhline(1,color='#777777',ls='--',lw=1)
    axes[0].set(xlabel='Number of friends m',ylabel='Actual gap / chord defect',ylim=(.6,1.03))
    axes[0].legend(fontsize=7,loc='lower right')
    points=plot['random_gap']
    axes[1].scatter([r['Delta'] for r in points],[r['exact_gap_interval'][1] for r in points],s=12,alpha=.55,color='#176b87')
    mx=max(r['Delta'] for r in points)*1.05
    axes[1].plot([0,mx],[0,mx],color='#777777',ls='--',lw=1)
    axes[1].set(xlabel='Computed chord defect',ylabel='Gap to feasible MILP margin',xlim=(-.005,mx),ylim=(-.003,mx))
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    figs=DEST/'figures';figs.mkdir(exist_ok=True)
    fig.savefig(figs/'gap-validation.png',dpi=210);plt.close(fig)
    g=gap['summary']
    lines=[r'\subsection{Verification and relaxation tightness}',
      f'The SAT construction was checked on all {sat["formulas"]} nonempty subsets of the eight three-variable clauses: {sat["satisfiable"]} satisfiable formulas and {sat["unsatisfiable"]} unsatisfiable formula. Exhaustive orthant LPs ({sat["piecewise_lps"]} total) agree with the sign-gap proof. Eight independent MILP checks also verify the raw and unit-box-scaled constructions. These finite tests check transcription; they do not replace the general complexity proof.',
      f'The real-subtable study contains {proto["object_milps"]} applicable object problems. All were numerically optimal, with maximum interval gap approximately $10^{{-8}}$. The pure and mixture bounds both certify {proto["onehot_certified"]} objects; the exact verifier certifies the same number. The pure bound is already exact to numerical precision on these instances.',
      r'\begin{table}[htbp]\centering\small',r'\caption{Fixed synthetic benchmark. Certification means a strictly positive worst-case margin. Improvement uses a $10^{-8}$ tolerance.}',
      r'\begin{tabular}{lrrrr}\toprule',r'Setting & Objects & Pure & Mixture & Bound improved\\\midrule']
    for group,data in bench['summary_by_group'].items():
        lines.append(f'{escape(group)} & {data["objects"]} & {data["onehot_certified"]} & {data["mixture_certified"]} & {data["strict_bound_improvement_count_tolerance_1e8"]}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:synthetic}\end{table}']
    lines += [r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{figures/gap-validation.png}',
        r'\caption{Left: the sharpness ratio $1-1/m$ for fixed-vertical and positive-width box families (the curves coincide). Right: retained feasible-MILP margin gaps versus the explicit chord defect on all 384 random objects. The dashed line is equality; it is not a fitted trend.}\label{fig:gap}\end{figure}',
        f'An independent chord-LP pass on the same {g["objects"]} random objects performs {g["enemy_chord_calls"]} enemy solves without new MILPs. No gap-bound inconsistency is observed. On the {g["zero_Delta_count"]} zero-defect objects, the largest feasible upper/lower interval is {g["max_zero_Delta_posterior_gap"]:.2g}. Mean chord defect is {g["mean_Delta"]:.6f}, compared with a mean a posteriori interval of {g["mean_posterior_gap"]:.6f} and mean actual numerical relaxation gap approximately {g["mean_exact_gap_interval"][1]:.6f}. The explicit bound is often conservative, as expected for a worst-case theorem. Ten constructed families use $m=3,5,9,17,33$, including positive-width boxes, and reproduce the sharpness ratios shown in Figure~\\ref{{fig:gap}}.']
    b=bench['summary']
    lines += [f'Across {b["objects"]} synthetic objects, mixtures tighten {b["strict_bound_improvement_count_tolerance_1e8"]} lower bounds but add no positive certifications. The average gap to the attained MILP value decreases from {b["mean_onehot_gap_interval_all"][1]:.8f} to {b["mean_mixture_gap_interval_all"][1]:.8f}. All {b["milp_optimal_count"]} MILPs report numerical optimality, with maximum retained interval gap $1.01\\times10^{{-8}}$. The bounds reveal {b["robust_status_counts"]["misclassification_witness"]} feasible misclassification witnesses and certify {b["mixture_certified"]} robust objects. Pure-bound, mixture and MILP computations take respectively {b["seconds"]["onehot_seconds"]:.3f}, {b["seconds"]["mixture_seconds"]:.3f} and {b["seconds"]["milp_seconds"]:.3f} seconds on this local run; these times are descriptive rather than hardware-independent complexity claims.',
      r'\subsection{First-fold attribute-selection pilot}',r'\begin{table}[htbp]\centering\small',
      r'\caption{All 15 first-fold results. Coverage is the fraction of full-table objects protected by the pure-witness certificate; the last column is the smallest class coverage.}',
      r'\begin{tabular}{lrrrr}\toprule Dataset & Full $d$ & Retained & Coverage & Min. class\\\midrule']
    for row in pilot['rows']:
        s=row['selected']
        lines.append(f'{escape(row["dataset"])} & {row["d"]} & {len(s["features"])} & {100*s["protected_fraction"]:.1f}\\% & {100*s["min_class_protected_fraction"]:.1f}\\%'+r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:datasets}\end{table}']
    s=pilot['summary']
    lines += [f'The mean deleted-attribute fraction is {100*s["mean_removed_fraction"]:.2f}\\%. Mean protected-object coverage is {100*s["mean_protected_fraction"]:.2f}\\%, and mean minimum-class coverage is {100*s["mean_min_class_protected_fraction"]:.2f}\\%. Ecoli, Glass, Vehicle and Yeast each contain at least one unprotected class; two Ecoli classes are singletons in this fold. Such results do not meet a requirement of positive coverage in every class. In particular, Yeast protects only about 3\\% of its training objects.',
      r'\begin{table}[htbp]\centering\small',r'\caption{Mean balanced accuracy (percent) across 15 datasets. The selection is fixed before held-out prediction. These are descriptive first-fold results.}',
      r'\begin{tabular}{llrrr}\toprule Classifier & Selection & Clean & Noise .01 & Noise .05\\\midrule']
    names={'all':'All attributes','certified_margin':'Certified margin','mi_matched':'Matched MI','nominal_margin':'Nominal margin'}
    for clf,label in [('l1_1nn','L1 1NN'),('euclidean_5nn','L2 5NN')]:
        for key in names:
            values=[next(v['mean_balanced_accuracy'] for v in s['predictive'] if v['classifier']==clf and v['method']==key and v['test_epsilon']==noise) for noise in [0.,.01,.05]]
            lines.append(label+' & '+names[key]+' & '+' & '.join(f'{100*v:.2f}' for v in values)+r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:prediction}\end{table}',
      'The certified subsets do not establish an advantage over all attributes or matched MI. The nominal ablation deletes using unperturbed margins and gives weaker predictive results here, but this observation is not a proof of statistical superiority for the robust selector. The full experiment stores 1320 method/classifier/perturbation records.']
    (DEST/'results.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('wrote bibliography and evidence-derived tables')
if __name__=='__main__':main()
