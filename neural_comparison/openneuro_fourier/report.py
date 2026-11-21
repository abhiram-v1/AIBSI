"""Matplotlib evidence for OpenNeuro frequency/diagnosis associations."""
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectral_data import HERE,CHANNELS,DESCRIPTORS,REGIONS,dump
from fourier_model import ARMS

NAMES={'amplitude':'Amplitude alone','summary':'Spectral summaries','relative_summary':'Relative/shape summaries',
       'absolute_spectrum':'Absolute Fourier spectrum','relative_spectrum':'Relative Fourier spectrum','selected':'Inner-selected pipeline'}
COLORS={'HC':'#207766','AD':'#8c3b49','FTD':'#b67d24'}


def save(fig,filename):
    fig.savefig(HERE/'outputs/figures'/filename,dpi=175,bbox_inches='tight');plt.close(fig)


def main():
    out=HERE/'outputs';figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    ps=json.loads((out/'prepared/plan.json').read_text())['participants'];arrays=dict(np.load(out/'prepared/features.npz'))
    association=json.loads((out/'associations.json').read_text());severity=json.loads((out/'severity_associations.json').read_text())
    dx=np.array([p['label'] for p in ps]);global_x=arrays['global_summary'];frequency=arrays['frequency']
    tests=association['families'];rng=np.random.default_rng(44)
    stats_lookup={contrast:{v['feature']:v for v in values} for contrast,values in tests['global'].items()}

    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,key,title in zip(axes,('psd','relative_psd'),('Absolute Fourier power','Relative Fourier power')):
        for d in ('HC','AD','FTD'):
            p=arrays['psd'][dx==d].mean(1)
            if key=='relative_psd':p=p/(p.sum(1,keepdims=True)*.25)
            median=np.median(p,axis=0);lo,hi=np.percentile(p,[25,75],axis=0)
            ax.plot(frequency,median,color=COLORS[d],label=f'{d} (n={len(p)})');ax.fill_between(frequency,lo,hi,color=COLORS[d],alpha=.12)
        ax.set_yscale('log');ax.set_xlabel('Frequency (Hz)');ax.set_title(title);ax.set_ylabel('µV²/Hz' if key=='psd' else 'Fraction of global1–40Hz power / Hz')
        ax.axvspan(4,8,color='#a98b39',alpha=.07);ax.axvspan(8,13,color='#437cad',alpha=.07);ax.legend(fontsize=9)
    fig.suptitle('OpenNeuro88: each curve summarizes participant spectra; shading is the interquartile range',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'fourier_group_spectra.png')

    fig,axes=plt.subplots(2,2,figsize=(13,10))
    panels=[('log_theta_alpha_ratio','Theta / alpha power','ratio'),('relative_alpha_8_13','Relative alpha power (8–13Hz)','percent'),
            ('alpha_peak_7_13_hz','Largest spectral peak within7–13Hz','hz'),('log_slow_fast_ratio','(Delta + theta) / (alpha + beta)','ratio')]
    for ax,(name,title,mode) in zip(axes.ravel(),panels):
        x=global_x[:,DESCRIPTORS.index(name)];x=10**x if mode=='ratio' else 100*x if mode=='percent' else x
        for i,d in enumerate(('HC','AD','FTD')):
            vals=x[dx==d];ax.boxplot(vals,positions=[i],widths=.55,showfliers=False,patch_artist=True,
                boxprops={'facecolor':COLORS[d],'alpha':.18},medianprops={'color':'black'})
            ax.scatter(i+rng.uniform(-.17,.17,len(vals)),vals,s=22,color=COLORS[d],alpha=.72)
        if mode=='ratio':ax.set_yscale('log')
        ax.set_xticks([0,1,2],['HC','AD','FTD']);v=stats_lookup['dementia_vs_HC'][name]
        ax.set_title(f'{title}\nDementia vs HC: r={v["point_biserial_r"]:.2f}, FDR q={v["fdr_q"]:.2g}',fontsize=11)
        ax.set_ylabel('Power ratio (log scale)' if mode=='ratio' else '% of1–40Hz power' if mode=='percent' else 'Hz')
    fig.suptitle('Frequency patterns are associated with diagnosis, but participant distributions overlap',fontsize=15)
    fig.text(.5,.015,'Dots =88 people, never independent EEG windows. Associations are exploratory full-cohort summaries; plots are not held-out performance estimates.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.95]);save(fig,'interpretable_frequency_patterns.png')

    contrast_keys=['AD_vs_HC','FTD_vs_HC','AD_vs_FTD'];fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True,sharey=True)
    size=19*len(frequency)
    for col,contrast in enumerate(contrast_keys):
        values=tests['frequency'][contrast]
        for row,kind in enumerate(('absolute','relative')):
            vv=values[row*size:(row+1)*size];r=np.array([v['point_biserial_r'] for v in vv]).reshape(19,-1)
            q=np.array([v['fdr_q'] for v in vv]).reshape(19,-1);ax=axes[row,col]
            im=ax.imshow(r,origin='upper',aspect='auto',cmap='RdBu_r',vmin=-.7,vmax=.7,
                         extent=[frequency[0]-.125,frequency[-1]+.125,18.5,-.5])
            mask=q<.05
            if mask.any() and not mask.all():ax.contour(frequency,np.arange(19),mask.astype(float),levels=[.5],colors='black',linewidths=.5)
            ax.set_yticks(np.arange(19),CHANNELS,fontsize=8);ax.set_title(f'{contrast.replace("_"," ")}: {kind} power',fontsize=11)
            if row==1:ax.set_xlabel('Frequency (Hz)')
    fig.subplots_adjust(top=.91,bottom=.10,wspace=.1,hspace=.18)
    c=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,pad=.025);c.set_label('Point-biserial r (positive = AD/FTD higher than HC; AD higher than FTD)')
    fig.suptitle('Which electrode frequencies associate with the diagnosis?',fontsize=16)
    fig.text(.5,.02,'Black outlines: individual-bin FDR q<.05 across5966 absolute+relative tests per contrast. Adjacent bins and electrodes are correlated; outlines are not cluster-level tests.',ha='center',fontsize=9)
    save(fig,'electrode_frequency_associations.png')

    contrasts=['dementia_vs_HC',*contrast_keys];fig,axes=plt.subplots(1,4,figsize=(16,8),sharey=True)
    band_names=DESCRIPTORS[5:10]
    for ax,contrast in zip(axes,contrasts):
        lookup={v['feature']:v for v in tests['channel_summary'][contrast]}
        vv=[lookup[f'{ch}:{band}'] for ch in CHANNELS for band in band_names]
        r=np.array([v['point_biserial_r'] for v in vv]).reshape(19,5)
        im=ax.imshow(r,cmap='RdBu_r',vmin=-.7,vmax=.7,aspect='auto');ax.set_title(contrast.replace('_',' '),fontsize=11)
        for i,v in enumerate(vv):
            if v['fdr_q']<.05:ax.text(i%5,i//5,'•',ha='center',va='center',color='black',fontsize=12)
        ax.set_xticks(range(5),['δ','θ','α','β','30–40']);ax.set_yticks(range(19),CHANNELS,fontsize=9)
    fig.subplots_adjust(top=.9,bottom=.1,wspace=.14)
    c=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,pad=.025);c.set_label('Point-biserial r for relative band power')
    fig.suptitle('Relative band-power associations across all19 electrodes',fontsize=16)
    fig.text(.5,.02,'Dots: FDR q<.05 within the361-feature channel-summary family per contrast. Statistical association does not establish a causal disease source.',ha='center',fontsize=10)
    save(fig,'electrode_band_associations.png')

    summaries={task:json.loads((out/f'evaluation/{task}_summary.json').read_text()) for task in ('binary','three_class')}
    fig,axes=plt.subplots(1,2,figsize=(17,6))
    for ax,(task,summary) in zip(axes,summaries.items()):
        arms=[*ARMS,'selected'];pos=np.arange(len(arms))
        for offset,key,color,label in [(-.14,'macro_f1','#3166a0','Macro F1'),(.14,'balanced_accuracy','#207766','Balanced accuracy')]:
            vals=np.array([summary[a]['mean'][key]*100 for a in arms]);bounds=np.array([summary[a]['bootstrap95'][key] for a in arms]).T*100
            ax.errorbar(pos+offset,vals,yerr=np.maximum(np.stack([vals-bounds[0],bounds[1]-vals]),0),fmt='o',capsize=3,color=color,label=label)
        ax.set_xticks(pos,[NAMES[a] for a in arms],rotation=30,ha='right');ax.set_ylim(15,100);ax.set_ylabel('Score (%)');ax.legend(fontsize=9)
        ax.set_title('HC versus AD+FTD' if task=='binary' else 'HC versus AD versus FTD')
    fig.suptitle('Does frequency analysis predict held-out participants?',fontsize=16)
    fig.text(.5,.015,'3repeats ×5 outer folds;3 inner folds. Feature ranking, feature count, scaling, SVM and binary threshold selected inside training folds. Bars: conditional participant bootstrap95% intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.93]);save(fig,'nested_svm_comparison.png')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(task,summary) in zip(axes,summaries.items()):
        cm=np.sum([v['confusion_matrix'] for v in summary['selected']['per_repeat']],axis=0);pct=100*cm/cm.sum(1,keepdims=True)
        ax.imshow(pct,cmap='Blues',vmin=0,vmax=100);labels=['HC','AD+FTD'] if task=='binary' else ['HC','AD','FTD']
        for i in range(len(cm)):
            for j in range(len(cm)):ax.text(j,i,f'{pct[i,j]:.1f}%',ha='center',va='center',fontsize=14,color='white' if pct[i,j]>60 else 'black')
        ax.set_xticks(range(len(cm)),labels);ax.set_yticks(range(len(cm)),labels);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
        ax.set_title(f'Inner-selected {"binary" if task=="binary" else "three-class"} pipeline')
    fig.suptitle('Row-normalized held-out predictions across3 repeats;88 distinct people',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'heldout_confusion_matrices.png')

    # The severity analysis is restricted to patients; controls all have MMSE30.
    fig,ax=plt.subplots(figsize=(8,5));j=DESCRIPTORS.index('log_theta_alpha_ratio')
    for d in ('AD','FTD'):
        ix=dx==d;ax.scatter(global_x[ix,j],[p['mmse'] for p in np.array(ps,dtype=object)[ix]],color=COLORS[d],label=d,alpha=.8)
    v=next(v for v in severity['tests'] if v['feature']=='global:log_theta_alpha_ratio')
    ax.set_xlabel('Log10 global theta / alpha power');ax.set_ylabel('MMSE score');ax.set_ylim(-1,31);ax.legend()
    ax.set_title(f'Within59 AD/FTD patients: severity association\nAge/sex/diagnosis-adjusted rank r={v["partial_rank_correlation"]:.2f}; FDR q={v["fdr_q"]:.2g}')
    fig.tight_layout();save(fig,'patient_severity_check.png')

    selections={task:Counter(v['arm'] for v in json.loads((out/f'evaluation/{task}_selection.json').read_text())) for task in summaries}
    top_features={}
    for task in summaries:
        counter=Counter()
        for path in (out/'evaluation').glob(f'{task}_[0-9]_[0-9].json'):
            rec=json.loads(path.read_text());entry=rec['arms'][rec['selected_arm']]
            for name in entry['selected_feature_names']:counter[name]+=1
        top_features[task]=[{'feature':k,'outer_fold_selected_count':v} for k,v in counter.most_common(40)]
    dump(out/'selected_feature_stability.json',top_features)
    # Log ratios are converted back only for descriptive medians, not model fitting.
    global_summary={}
    for name in DESCRIPTORS:
        vals=global_x[:,DESCRIPTORS.index(name)]
        global_summary[name]={d:{'median':float(np.median(vals[dx==d])),'q25_q75':np.percentile(vals[dx==d],[25,75]).tolist()} for d in ('HC','AD','FTD')}
    dump(out/'global_marker_distributions.json',global_summary)
    lines=['# OpenNeuro Fourier analysis: finding diagnosis-associated patterns','',
        '**There are measurable frequency differences in this cohort.** Overall amplitude is a weak discriminator, while relative alpha, theta/alpha ratio and slowing are associated with HC versus AD/FTD. These groups still overlap, and AD-versus-FTD differences are much weaker.',
        '', '## What the Fourier analysis does','',
        'An EEG channel contains a mixture of activity over frequencies and sources. FFT/Welch estimates its frequency power. It does not uniquely unmix brain sources, identify a particular neuron population, or prove that a frequency component is caused by disease. STFT in earlier experiments already used Fourier transforms; this experiment explicitly measures label associations and performs fold-internal frequency feature selection.',
        '', 'All88 OpenNeuro people are included:36 AD,23 FTD,29 HC. We use supplied cleaned derivatives, the same256Hz/19-channel CAR/1–40Hz FIR and accepted8s windows as the compatibility audit. At most40 windows/person. Welch:4-second Hann periodograms,2-second hop,1024-point FFT;157 frequency bins at0.25Hz spacing from1–40Hz. The delta band here is1–4Hz because this processing policy excludes frequencies below1Hz.',
        '', 'The averaged PSD has shape88×19×157. Flattened Fourier features give88×2983 matrices. Spectral summaries give88×361, relative/shape summaries88×247, and the amplitude control88×1. Window spectra are aggregated within each person; windows are never counted as independent participants.',
        '', 'Summary features: absolute and relative power in five bands; slow/fast and theta/alpha ratios;7–13Hz peak location; spectral centroid, entropy and95% edge; total power; a rough background slope and alpha prominence. The background fit is a log-log straight-line proxy, not a validated oscillation/aperiodic decomposition. A7–13Hz peak location is the largest bin in that range, not proof of a distinct alpha oscillation.',
        '', '## Which patterns correlate with labels?','',
        'Global spectra average channels within each participant. The following are descriptive full-cohort comparisons; they were not used to choose classifier features.',
        '', '| Global marker | HC median | AD median | FTD median | Dementia–HC r | FDR q | Age/sex-adjusted q |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ['log_theta_alpha_ratio','relative_alpha_8_13','relative_theta_4_8','log_slow_fast_ratio','alpha_peak_7_13_hz','alpha_above_background_db']:
        med=[global_summary[name][d]['median'] for d in ('HC','AD','FTD')];suffix=''
        if name in ('log_theta_alpha_ratio','log_slow_fast_ratio'):
            med=[float(np.median(10**global_x[dx==d,DESCRIPTORS.index(name)])) for d in ('HC','AD','FTD')]
            label=name.replace('log_','')+' (power ratio)'
        elif name.startswith('relative'):med=[100*v for v in med];label=name+' (%)'
        else:label=name
        v=stats_lookup['dementia_vs_HC'][name]
        lines.append('|'+label+'|'+'|'.join(f'{v:.3g}' for v in med)+f'|{v["point_biserial_r"]:.3f}|{v["fdr_q"]:.3g}|{v["adjusted_fdr_q"]:.3g}|')
    lines+=['','Power-ratio medians are displayed on the original scale; ratio correlations/tests use their log10 features. Positive r means higher values in the patient group; negative r means lower. Categorical diagnoses were not correlated as arbitrary0/1/2 codes. We tested HC versus pooled AD+FTD and each explicit pairwise contrast. Mann–Whitney tests and age/sex-adjusted OLS results are both saved. Age/sex adjustment is exploratory, assumes the specified linear model, and does not eliminate unmeasured confounding.',
        '', 'FDR is corrected separately within each declared feature family and contrast: global19, regional76, channel-summary361, and combined absolute+relative electrode-frequency5966 tests. These are related exploratory views, not independent replications. Adjacent frequencies and electrodes are correlated; a large number of significant bins is not a large number of independent biomarkers.',
        '', '| Contrast | Global markers q<.05 /19 | Channel summaries q<.05 /361 | Frequency values q<.05 /5966 |','|---|---:|---:|---:|']
    for contrast in ('dementia_vs_HC','AD_vs_HC','FTD_vs_HC','AD_vs_FTD'):
        counts=[sum(v['fdr_q']<.05 for v in tests[f][contrast]) for f in ('global','channel_summary','frequency')]
        lines.append('|'+contrast+'|'+'|'.join(map(str,counts))+'|')
    lines+=['','No AD-versus-FTD feature survived these FDR thresholds. This is insufficient evidence for a difference under this analysis; it does not establish that AD and FTD have identical EEG physiology.',
        '', '## Do these features classify held-out participants?','',
        '3 repeated5-fold outer splits, stratified by original diagnosis;3-fold inner selection. ANOVA feature ranking, number of features (5/15/40/all), scaling, linear/RBF SVM C(.01/.1/1/10), and binary score threshold(−.5/0/.5) are fitted/selected inside training folds. Balanced class weights are used. There is no PCA. Age,sex,MMSE,participant IDs, and recording length are not classifier features.',
        '', '“Inner-selected” chooses the feature family using inner-fold scores. It is the estimate for the complete selection procedure. Picking the highest outer-CV family afterward gives a descriptive ablation result, not a new independently validated optimum.',
        '', '| Method | Binary F1 | Binary balanced accuracy | Three-class F1 | Three-class balanced accuracy |','|---|---:|---:|---:|---:|']
    for arm in (*ARMS,'selected'):
        b=summaries['binary'][arm]['mean'];t=summaries['three_class'][arm]['mean']
        lines.append(f'|{NAMES[arm]}|{100*b["macro_f1"]:.1f}%|{100*b["balanced_accuracy"]:.1f}%|{100*t["macro_f1"]:.1f}%|{100*t["balanced_accuracy"]:.1f}%|')
    lines+=['','The amplitude-only binary score exactly reproduces the earlier OpenNeuro control. A Fourier spectrum therefore provides useful information beyond overall amplitude in this cohort, while statistical association remains compatible with substantial participant-level overlap.',
        '', f'Inner-selected feature-family counts across15 outer folds: binary{dict(selections["binary"])}, three-class{dict(selections["three_class"])}.',
        '', 'Intervals are participant bootstraps that keep all repeated predictions for each person together. They are conditional on fitted models, not full retraining intervals. Repeats reuse the same88 people. This cohort has informed many earlier project experiments, so results are exploratory rather than an untouched external evaluation.',
        '', '## Clinical labels and MMSE','',
        'The original descriptor reports that initial AD/FTD diagnoses used DSM-IIIR/DSM-IV/ICD-10 and NINCDS-ADRDA criteria, with MMSE to characterize cognitive status. These are clinical labels; the descriptor does not define them by an EEG amplitude or Fourier threshold. We cannot infer the availability of every imaging/CSF/pathology test for every individual from this release. Source: [Miltiadous et al., Data2023,8,95](https://doi.org/10.3390/data8060095), section2.1/page3; retrieval/hash recorded in clinical_context.json.',
        '', 'MMSE is excluded from classification. Controls all score30, so all-cohort MMSE correlation would partly repeat the diagnosis contrast. We instead analyzed only59 AD/FTD patients, with age/sex/AD-versus-FTD adjustment of rank correlations. No severity association survived FDR correction in the380 global/channel-summary tests; the minimum q is'+f'{min(v["fdr_q"] for v in severity["tests"]):.3f}.',
        '', '## What this establishes','',
        'OpenNeuro contains label-associated spectral structure, including relatively weaker alpha and greater theta/alpha ratio in the patient groups. Those patterns provide moderate held-out binary classification, with weaker AD/FTD discrimination. The analysis does not show a universally separable disease signature, establish biological causation, or contradict clinical diagnoses when an EEG-only model makes errors.',
        '', 'Earlier conventional spectral features and log-mel/STFT inputs already represented related information. This is a clearer analysis of which features correlate with labels, not evidence that Fourier transformation is a previously unused source-separation method.',
        '', '## Outputs','',
        '- `prepared/features.npz`, `feature_names.json`: participant matrices and exact feature names.',
        '- `associations.json`: per-feature effects, correlations, raw/adjusted p and FDR q.',
        '- `severity_associations.json`: within-patient severity analysis.',
        '- `evaluation/protocol.json`, source snapshots, folds and predictions: reproducible nested CV.',
        '- `selected_feature_stability.json`: feature selection counts across outer training folds; selection frequency is not causal importance.',
        '- `figures/*.png`: seven Matplotlib figures; no PDF/CSV.','']
    for name in ('fourier_group_spectra','interpretable_frequency_patterns','electrode_frequency_associations','electrode_band_associations',
                 'nested_svm_comparison','heldout_confusion_matrices','patient_severity_check'):
        lines.append(f'![{name}]({(figdir/(name+".png")).as_posix()})\n')
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Detailed report and seven Matplotlib figures saved.',flush=True)


if __name__=='__main__':main()
