"""Matplotlib cohort audit, paired evaluation figures, and a reproducible report."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from audit_data import HERE,CHANNELS,dump
from compat_model import metrics

OUT=HERE/'outputs';PREP=OUT/'prepared';EV=OUT/'evaluation';FIG=OUT/'figures'
COLORS=['#3166a0','#c56a32'];NAMES=['OpenNeuro','BrainLat']
LABELS={'raw':'Existing derivatives','common':'Matched CAR + filter','gain':'Matched + gain norm.',
        'relative':'Matched + relative power','site_z':'Matched + site z-score',
        'site_coral':'Matched + site CORAL','selected':'Inner-selected method'}


def save(fig,name):
    fig.savefig(FIG/name,dpi=170,bbox_inches='tight');plt.close(fig)


def fdr(p):
    p=np.asarray(p);o=np.argsort(p);q=np.empty(len(p));q[o]=np.minimum(1,np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q


def effect(a,b):
    # Positive = BrainLat higher; Hedges g and rank-biserial/Cliff delta.
    a=np.asarray(a);b=np.asarray(b);n=len(a)+len(b)
    sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(n-2))
    u,p=mannwhitneyu(b,a,alternative='two-sided')
    return {'openneuro_n':len(a),'brainlat_n':len(b),'openneuro_median':float(np.median(a)),
            'brainlat_median':float(np.median(b)),'hedges_g':float((b.mean()-a.mean())/max(sd,1e-15)*(1-3/(4*n-9))),
            'cliff_delta':float(2*u/(len(a)*len(b))-1),'mann_whitney_p':float(p)}


def domain_bounds(y,p):
    rng=np.random.default_rng(2026110588);groups=[np.flatnonzero(y==k) for k in (0,1)];values=[]
    for _ in range(1000):
        ix=[rng.choice(g,len(g),replace=True) for g in groups]
        values.append(np.mean([np.mean(p[:,ix[k]]==k) for k in (0,1)]))
    return np.percentile(values,[2.5,97.5]).tolist()


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    ps=json.loads((PREP/'plan.json').read_text())['participants'];n=len(ps)
    meta=[json.loads((PREP/'parts'/f'{i:03d}.json').read_text()) for i in range(n)]
    arrays=dict(np.load(PREP/'features.npz'));domain=np.array([p['source']=='brainlat' for p in ps],int)
    dx=np.array([p['label'] for p in ps]);sites=np.array([p['site'] for p in ps]);ages=np.array([np.nan if p['age'] is None else p['age'] for p in ps])
    keys=['median_channel_rms_uv','common_mode_variance_fraction','median_pair_correlation','median_excess_kurtosis',
          'high_frequency_30_40_fraction','residual_line50_ratio','residual_line60_ratio']
    values={stage:{key:np.array([m[f'{stage}_metrics'][key] for m in meta]) for key in keys} for stage in ('raw','common')}
    audit={'participants':n,'cohort_counts':{name:{d:int(np.sum((domain==s)&(dx==d))) for d in ('AD','FTD','HC')} for s,name in enumerate(NAMES)},
           'site_counts':{s:{d:int(np.sum((sites==s)&(dx==d))) for d in ('AD','FTD','HC')} for s in np.unique(sites)},
           'windows':{'total':sum(m['windows']['accepted_windows'] for m in meta),'minimum':min(m['windows']['accepted_windows'] for m in meta),
                      'maximum':max(m['windows']['accepted_windows'] for m in meta),'unavailable_brainlat_fdt_files':14},
           'cohort_signal_summary':{},'within_diagnosis_signal_tests':[],'age':{},'channel_band_tests':[]}
    for stage in values:
        audit['cohort_signal_summary'][stage]={name:{k:{'median':float(np.median(v[domain==s])),
                'q25_q75':np.percentile(v[domain==s],[25,75]).tolist()} for k,v in values[stage].items()} for s,name in enumerate(NAMES)}
        for d in ('AD','FTD','HC'):
            for k,v in values[stage].items():
                audit['within_diagnosis_signal_tests'].append({'stage':stage,'diagnosis':d,'metric':k,
                    **effect(v[(domain==0)&(dx==d)],v[(domain==1)&(dx==d)])})
    q=fdr([v['mann_whitney_p'] for v in audit['within_diagnosis_signal_tests']])
    for v,a in zip(audit['within_diagnosis_signal_tests'],q):v['fdr_q']=float(a)
    for d in ('AD','FTD','HC'):
        audit['age'][d]=effect(ages[(domain==0)&(dx==d)&np.isfinite(ages)],ages[(domain==1)&(dx==d)&np.isfinite(ages)])
    band_arrays={'raw':np.log10(np.maximum(arrays['raw_bandpower'],1e-20)),
                 'common':np.log10(np.maximum(arrays['common_bandpower'],1e-20)),
                 'relative':np.log10(np.maximum(arrays['common_bandpower']/arrays['common_bandpower'].sum(2,keepdims=True),1e-20))}
    for stage,x in band_arrays.items():
        for d in ('AD','FTD','HC'):
            for c,ch in enumerate(CHANNELS):
                for b,band in enumerate(['delta','theta','alpha','beta','30–40Hz']):
                    audit['channel_band_tests'].append({'stage':stage,'diagnosis':d,'channel':ch,'band':band,
                        **effect(x[(domain==0)&(dx==d),c,b],x[(domain==1)&(dx==d),c,b])})
    q=fdr([v['mann_whitney_p'] for v in audit['channel_band_tests']])
    for v,a in zip(audit['channel_band_tests'],q):v['fdr_q']=float(a)
    dump(OUT/'signal_audit.json',audit)

    # No epoch pseudo-replication: every plotted point is one person.
    rng=np.random.default_rng(42);fig,axes=plt.subplots(2,3,figsize=(17,9))
    panels=[('median_channel_rms_uv','Median channel RMS (µV)',True),
            ('common_mode_variance_fraction','Common-mode variance / mean channel variance',False),
            ('median_pair_correlation','Median inter-channel correlation',False)]
    for ax,(key,title,log) in zip(axes[0],panels):
        for stage_i,stage in enumerate(('raw','common')):
            for s in (0,1):
                v=values[stage][key][domain==s];pos=stage_i*3+s
                ax.boxplot(v,positions=[pos],widths=.65,showfliers=False,patch_artist=True,
                           boxprops={'facecolor':COLORS[s],'alpha':.35},medianprops={'color':'black'})
                ax.scatter(pos+rng.uniform(-.2,.2,len(v)),v,s=10,color=COLORS[s],alpha=.5)
        ax.set_xticks([0,1,3,4],['OpenNeuro\nexisting','BrainLat\nexisting','OpenNeuro\nmatched','BrainLat\nmatched'],fontsize=9)
        ax.set_title(title)
        if log:ax.set_yscale('log')
    ax=axes[1,0]
    for j,d in enumerate(('AD','FTD','HC')):
        for s in (0,1):
            v=ages[(dx==d)&(domain==s)&np.isfinite(ages)];pos=j*3+s
            ax.boxplot(v,positions=[pos],widths=.65,showfliers=False,patch_artist=True,boxprops={'facecolor':COLORS[s],'alpha':.35},medianprops={'color':'black'})
            ax.scatter(pos+rng.uniform(-.18,.18,len(v)),v,s=12,color=COLORS[s],alpha=.6)
    ax.set_xticks([.5,3.5,6.5],['AD','FTD','HC']);ax.set_title('Age differs within diagnosis');ax.set_ylabel('Years')
    ax=axes[1,1]
    for j,s in enumerate(('AHEPA','AR','CL')):
        for k,d in enumerate(('AD','FTD','HC')):
            v=values['common']['median_channel_rms_uv'][(sites==s)&(dx==d)];pos=j*4+k
            ax.scatter(pos+rng.uniform(-.2,.2,len(v)),v,s=15,alpha=.6,color=['#8c3b49','#996b16','#207766'][k],label=d if j==0 else None)
            ax.plot([pos-.25,pos+.25],[np.median(v)]*2,color='black',linewidth=2)
    ax.set_xticks([1,5,9],['OpenNeuro\nAHEPA','BrainLat\nArgentina','BrainLat\nChile']);ax.set_yscale('log');ax.set_ylabel('RMS (µV)');ax.set_title('Residual gain differences after matching');ax.legend(fontsize=9)
    ax=axes[1,2]
    for s in (0,1):
        ix=domain==s;ax.scatter(values['raw']['median_channel_rms_uv'][ix],values['common']['median_channel_rms_uv'][ix],label=NAMES[s],color=COLORS[s],alpha=.65,s=22)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Existing RMS (µV)');ax.set_ylabel('Matched RMS (µV)');ax.set_title('Paired change for each participant');ax.legend()
    fig.suptitle('Cohort compatibility audit: 88 OpenNeuro + 86 usable BrainLat participants',fontsize=16)
    fig.text(.5,.01,'“Existing” = supplied derivatives mapped to19 channels and resampled. “Matched” adds the same19-channel CAR and1–40Hz filter. Dots = people.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95]);save(fig,'signal_differences.png')

    fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True);freq=np.arange(1,40.01,.5)
    for row,stage in enumerate(('raw','common')):
        for col,d in enumerate(('AD','FTD','HC')):
            ax=axes[row,col]
            for s in (0,1):
                v=arrays[f'{stage}_profile'][(dx==d)&(domain==s)]
                ax.plot(freq,np.median(v,axis=0),color=COLORS[s],label=f'{NAMES[s]} (n={len(v)})')
                lo,hi=np.percentile(v,[25,75],axis=0);ax.fill_between(freq,lo,hi,color=COLORS[s],alpha=.14)
            ax.set_yscale('log');ax.set_title(f'{d}: {"existing derivatives" if row==0 else "matched CAR + filter"}');ax.set_ylabel('Power density (µV²/Hz)');ax.legend(fontsize=9)
            if row:ax.set_xlabel('Frequency (Hz)')
    fig.suptitle('Same windows, separate diagnoses: median spectra and interquartile ranges',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.95]);save(fig,'spectral_comparison.png')

    fig,axes=plt.subplots(3,3,figsize=(15,16),sharex=True,sharey=True)
    for row,stage in enumerate(band_arrays):
        for col,d in enumerate(('AD','FTD','HC')):
            tests=[v for v in audit['channel_band_tests'] if v['stage']==stage and v['diagnosis']==d]
            g=np.array([v['hedges_g'] for v in tests]).reshape(19,5);ax=axes[row,col]
            image=ax.imshow(g,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto')
            for i,v in enumerate(tests):
                if v['fdr_q']<.05:ax.text(i%5,i//5,'•',ha='center',va='center',color='black',fontsize=9)
            ax.set_title(f'{d}: {stage}');ax.set_xticks(range(5),['δ','θ','α','β','30–40']);ax.set_yticks(range(19),CHANNELS,fontsize=9)
    fig.subplots_adjust(top=.93,bottom=.09,hspace=.14,wspace=.16);cb=fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.6,pad=.025)
    cb.set_label('Hedges g of log10 power (positive = BrainLat higher; clipped at ±2)')
    fig.suptitle('Residual channel/band differences within the same diagnosis',fontsize=16)
    fig.text(.5,.025,'Dots: Mann–Whitney FDR q<.05 across all855 comparisons. Relative = channel band power / channel1–40Hz power. Differences are not proof of noise.',ha='center',fontsize=10)
    save(fig,'channel_band_differences.png')

    if not (EV/'summary.json').exists():
        print('Signal report figures ready; rerun after evaluation completes for model figures.',flush=True);return
    summary=json.loads((EV/'summary.json').read_text());pred=dict(np.load(EV/'predictions.npz'))
    subgroup={}
    for arm in summary:
        subgroup[arm]={}
        for group,mask in [('OpenNeuro',domain==0),('BrainLat',domain==1),*[(s,sites==s) for s in np.unique(sites)]]:
            runs=[metrics(pred['y'][mask],p[mask]) for p in pred[f'diagnosis_{arm}']]
            subgroup[arm][group]={k:float(np.mean([m[k] for m in runs])) for k in runs[0] if k!='confusion_matrix'}
    dump(OUT/'subgroup_metrics.json',subgroup)
    probe_ci={a:{k:domain_bounds(domain,pred[f'domain_{a}_{k}']) for k in ('linear','rbf')} for a in summary if a!='selected'}
    dump(OUT/'domain_probe_intervals.json',probe_ci)
    arms=list(summary);fig,axes=plt.subplots(1,2,figsize=(17,6))
    pos=np.arange(len(arms));ax=axes[0]
    for offset,key,color,name in [(-.16,'macro_f1','#3166a0','Macro F1'),(.16,'balanced_accuracy','#207766','Balanced accuracy')]:
        v=np.array([summary[a]['mean'][key]*100 for a in arms]);bounds=np.array([summary[a]['conditional_bootstrap95'][key] for a in arms]).T*100
        ax.errorbar(pos+offset,v,yerr=np.maximum(np.stack([v-bounds[0],bounds[1]-v]),0),fmt='o',color=color,label=name,capsize=3)
    ax.set_xticks(pos,[LABELS[a] for a in arms],rotation=35,ha='right');ax.set_ylabel('Score (%)');ax.set_ylim(40,100);ax.set_title('Binary diagnosis: repeated nested participant CV');ax.legend()
    ax=axes[1]
    for offset,k,color in [(-.14,'linear','#996b16'),(.14,'rbf','#8c3b49')]:
        aa=arms[:-1];v=np.array([summary[a]['domain_balanced_accuracy'][k]*100 for a in aa]);bounds=np.array([probe_ci[a][k] for a in aa]).T*100
        ax.errorbar(np.arange(len(aa))+offset,v,yerr=np.maximum(np.stack([v-bounds[0],bounds[1]-v]),0),fmt='o',color=color,label=f'{k.upper()} dataset probe',capsize=3)
    ax.axhline(50,linestyle='--',color='gray',label='Chance');ax.set_xticks(np.arange(len(aa)),[LABELS[a] for a in aa],rotation=35,ha='right')
    ax.set_ylim(35,102);ax.set_ylabel('Dataset-prediction balanced accuracy (%)');ax.set_title('Can a held-out person’s dataset still be recognized?');ax.legend(fontsize=9)
    fig.suptitle('Better matching must preserve diagnosis information',fontsize=16)
    fig.text(.5,.01,'3 repeats ×5 outer folds;3 inner folds. PCA20 and all site corrections fit inside training folds. Bars: participant bootstrap95% intervals, conditional on these fitted models.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);save(fig,'compatibility_vs_diagnosis.png')

    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,a in zip(axes,['raw','common','selected']):
        cm=np.sum([v['confusion_matrix'] for v in summary[a]['per_repeat']],axis=0);pct=100*cm/cm.sum(1,keepdims=True)
        ax.imshow(pct,vmin=0,vmax=100,cmap='Blues')
        for i in (0,1):
            for j in (0,1):ax.text(j,i,f'{pct[i,j]:.1f}%',ha='center',va='center',fontsize=15,color='white' if pct[i,j]>60 else 'black')
        ax.set_xticks([0,1],['HC','Dementia']);ax.set_yticks([0,1],['HC','Dementia']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(LABELS[a])
    fig.suptitle('Row-normalized held-out predictions; each person tested once per repeat',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.9]);save(fig,'confusion_matrices.png')

    transfers=json.loads((EV/'transfer.json').read_text());fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,t in zip(axes,transfers):
        aa=list(t['arms']);xx=np.arange(len(aa))
        for off,key,name,color in [(-.18,'macro_f1','Macro F1','#3166a0'),(.18,'balanced_accuracy','Balanced accuracy','#207766')]:
            bars=ax.bar(xx+off,[100*t['arms'][a]['metrics'][key] for a in aa],.34,label=name,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=9)
        ax.axhline(50,linestyle='--',color='gray',linewidth=.8,label='50% balanced accuracy');ax.set_ylim(0,105)
        ax.set_xticks(xx,[LABELS[a] for a in aa],rotation=25,ha='right');ax.set_title(f'Train {t["source"]} → test {"BrainLat" if t["source"]=="OpenNeuro" else "OpenNeuro"}\nSource-CV choice: {LABELS[t["selected_using_source_only"]]}',fontsize=12)
        ax.set_ylabel('Score (%)');ax.legend(fontsize=9)
    fig.suptitle('Entire other cohort held outside model selection and normalization',fontsize=15)
    fig.text(.5,.015,'Every method’s SVM is tuned on the source only. Primary method selected using source CV only. Both cohorts were explored previously; this is a transfer audit.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'cross_dataset_transfer.png')

    follow=OUT/'followup';native=[];amplitude={}
    if (follow/'amplitude_probe.json').exists():
        native=json.loads((follow/'native_metadata.json').read_text());amplitude=json.loads((follow/'amplitude_probe.json').read_text())
        fig,axes=plt.subplots(1,3,figsize=(17,5.5));rng=np.random.default_rng(42)
        ax=axes[0]
        for j,site in enumerate(('AHEPA','AR','CL')):
            for k,d in enumerate(('AD','FTD','HC')):
                v=[p['native_car_rms_uv_fixed56s'] for p in native if p['site']==site and p['label']==d];pos=j*4+k
                ax.scatter(pos+rng.uniform(-.2,.2,len(v)),v,s=15,alpha=.7,color=['#8c3b49','#996b16','#207766'][k],label=d if j==0 else None)
                ax.plot([pos-.25,pos+.25],[np.median(v)]*2,color='black',linewidth=2)
        ax.set_xticks([1,5,9],['OpenNeuro\n19 original','BrainLat AR\n128 original','BrainLat CL\n128 original']);ax.set_yscale('log')
        ax.set_ylabel('Median native-channel RMS (µV)');ax.set_title('Amplitude gap exists before interpolation');ax.legend(fontsize=9)
        ax=axes[1];versions=('2020.0','2022.1');table=np.array([[sum(p['source']=='brainlat' and p['label']==d and p['header_eeglab_version']==v for p in native) for v in versions] for d in ('AD','FTD','HC')])
        ax.imshow(table,cmap='Blues',vmin=0,vmax=35,aspect='auto')
        for i in range(3):
            for j in range(2):ax.text(j,i,str(table[i,j]),ha='center',va='center',fontsize=20,color='white' if table[i,j]>20 else 'black')
        ax.set_xticks([0,1],versions);ax.set_yticks(range(3),['AD','FTD','HC']);ax.set_xlabel('EEGLAB version recorded in .set metadata');ax.set_title('BrainLat version and diagnosis align')
        ax=axes[2];groups=['OpenNeuro','BrainLat','Pooled'];v=[100*amplitude[g]['mean']['macro_f1'] for g in groups]
        bars=ax.bar(range(3),v,color=[COLORS[0],COLORS[1],'#666666']);ax.bar_label(bars,fmt='%.1f%%',padding=3)
        ax.set_xticks(range(3),groups);ax.set_ylim(0,110);ax.set_ylabel('Macro F1 (%)');ax.set_title('Diagnosis from one RMS feature alone')
        fig.suptitle('Exploratory follow-up: amplitude and processing/export provenance need investigation',fontsize=15)
        fig.text(.5,.015,'Native audit: fixed 10–70s segment, native CAR +1–40Hz, trim2s edges. RMS probe:3×5 outer/3 inner CV. Recorded version does not prove the cause of the amplitude difference.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.07,1,.92]);save(fig,'amplitude_and_version_warning.png')

    lines=['# OpenNeuro–BrainLat compatibility audit','',
        'This audit compares existing EEG derivatives, before and after a matched processing stage. It cannot reconstruct original unprocessed signals or undo earlier ICA/ASR/interpolation.',
        '',f'Fixed cohort: **{n} participants** (88 OpenNeuro,86 BrainLat);71 AD,42 FTD,61 HC. BrainLat comprises Argentina48 and Chile38.14 additional HC headers lack their external EEG data files and cannot be analyzed.',
        '',f'Common rules: same19 channels (existing legacy spline operator for BrainLat), anti-aliased256Hz resampling,19-channel average reference (CAR),1–40Hz FIR,8-second windows, edge/boundary guards,0.1–200µV SD and500µV amplitude limits. All174 pass;{audit["windows"]["total"]} retained windows, at most40/person. Existing and matched signals use the exact same accepted windows.',
        '', 'STFT:2-second Hann,0.5-second hop,24 mel bands. Per-person frame mean/SD gives174×912 matrices. These are participant samples, not independent window samples.',
        '', '**Main finding:** signal matching is possible, but the current derivatives are not demonstrably interchangeable. A classifier still identifies the dataset with90.6% balanced accuracy after matched CAR/filtering. Stronger site alignment reduces detectable dataset differences but also lowers diagnosis performance.',
        '', '## Measured signal differences','',
        '| Metric (median across people) | OpenNeuro existing | BrainLat existing | OpenNeuro matched | BrainLat matched |','|---|---:|---:|---:|---:|']
    for key in keys:
        v=[audit['cohort_signal_summary'][st][name][key]['median'] for st in ('raw','common') for name in NAMES]
        lines.append('|'+key+'|'+'|'.join(f'{x:.5g}' for x in v)+'|')
    lines+=['','Native sample rates are500Hz for OpenNeuro and512Hz for BrainLat, now both resampled to256Hz. File metadata confirm reference `A1 A2` for all88 OpenNeuro files and `average` for all86 BrainLat files. Matching the reference is a substantive change. BrainLat channel values are interpolated estimates; the original19 OpenNeuro electrode measurements cannot be made equivalent to them by schema conversion alone.',
        '', 'The common-mode fraction measures variance shared in the channel average. It is consistent with a major reference/shared-signal difference; it does **not** prove that shared activity is artifact. CAR mathematically removes the channel mean, so near-zero common-mode variance afterward is expected, not validation of biological equivalence. All voltage figures use the EEGLAB/MNE microvolt convention; independent acquisition gain/calibration has not been verified from these headers.',
        '', 'Median age by diagnosis (OpenNeuro→BrainLat): '+', '.join(f'{d} {audit["age"][d]["openneuro_median"]:.0f}→{audit["age"][d]["brainlat_median"]:.0f}' for d in ('AD','FTD','HC'))+' years.6 BrainLat participants lack unambiguous matched age metadata. Age, severity, population and acquisition differences can coexist. MMSE and MoCA are not treated as interchangeable scores.',
        '', 'Channel/band tests are stratified by AD/FTD/HC, with FDR correction across855 tests. Significant differences do not by themselves identify bad electrodes or justify deleting channels. The age/diagnosis plots and effect maps are descriptive, not feature-selection input.',
        '', '## Does matching preserve diagnosis performance?','',
        'Binary task: HC versus AD+FTD.3 repeated5-fold outer participant splits, stratified by site×original diagnosis;3-fold inner selection. All six transformations are refitted inside each inner/outer training fold. SVM: linear/RBF, C=.01/.1/1/10, balanced class weights, decision thresholds−.5/0/.5. Global standardization and PCA20 are shared controls. Hyperparameter selection maximizes pooled inner macro F1, then balanced accuracy.',
        '', '| Method | Macro F1 | Balanced accuracy | Accuracy | HC recall | Dementia recall | Dataset probe BA (linear/RBF) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,v in summary.items():
        m=v['mean'];dp=v.get('domain_balanced_accuracy');dtext=f'{100*dp["linear"]:.1f}/{100*dp["rbf"]:.1f}' if dp else '—'
        lines.append('|'+LABELS[a]+'|'+'|'.join(f'{100*m[k]:.1f}%' for k in ('macro_f1','balanced_accuracy','accuracy','healthy_recall','dementia_recall'))+'|'+dtext+'|')
    lines+=['','These are **binary** internal results and are not comparable to earlier three-class results as a claim of improvement. Existing-derivative pooled macro F1 is82.4%, but the same held-out predictions give '+f'{100*subgroup["raw"]["OpenNeuro"]["macro_f1"]:.1f}% within OpenNeuro and{100*subgroup["raw"]["BrainLat"]["macro_f1"]:.1f}% within BrainLat. See subgroup_metrics.json.',
        '', '“Inner-selected” chooses the processing method as well as SVM settings using only the inner folds. It is the fair estimate of the method-selection procedure; do not select the best outer result and call that a new validated optimum.',
        '', 'Site z-score uses training-site feature means/SD. CORAL starts from the gain-normalized features and aligns training-site PCA-space means/covariances to the pooled training covariance, with30% covariance shrinkage. These are distinct arms, not sequential corrections. No test labels, means or covariances enter a fit. Known-site corrections assume site metadata at inference; they cannot estimate an unseen site from zero observations.',
        '', 'Dataset probes predict OpenNeuro versus BrainLat on held-out people.50% balanced accuracy is chance. Diagnosis-conditioned probes are also in summary.json. They fit probes separately within each diagnosis as a descriptive check; diagnosis is never given to a deployed harmonizer. A probe near chance only establishes reduced detectable site information for this feature/PCA/probe setup, not that the cohorts are biologically identical. Some diagnosis-conditioned CORAL probes score below50%; reversed associations or overcorrection can produce this, so those scores do not demonstrate that dataset information is absent.',
        '', '## Source-only cross-dataset transfer','',
        '| Training cohort | Method selected in source CV | Target macro F1 | Target balanced accuracy | HC recall | Dementia recall |',
        '|---|---|---:|---:|---:|---:|']
    for t in transfers:
        a=t['selected_using_source_only'];m=t['arms'][a]['metrics']
        lines.append('|'+t['source']+'|'+LABELS[a]+'|'+'|'.join(f'{100*m[k]:.1f}%' for k in ('macro_f1','balanced_accuracy','healthy_recall','dementia_recall'))+'|')
    lines+=['','Only deterministic per-record corrections are eligible for transfer; no target-cohort normalization or parameter selection. All other method results are saved as diagnostic ablations and plotted; choosing one using target performance would invalidate the source-only selection claim.',
        '', '## Exploratory amplitude/provenance follow-up','']
    if native:
        lines+=['This follow-up was added after observing the diagnosis-specific amplitude separation in the main audit; it is not an independent confirmatory endpoint.',
            '', '**All54 BrainLat AD/FTD files record EEGLAB2020.0, and all32 usable BrainLat HC files record EEGLAB2022.1.** OpenNeuro records2022.1 for every diagnosis. This is a perfect association between the recorded BrainLat processing/export version and the binary label. A version field alone does not establish that algorithms, settings or gains changed, and does not prove label leakage or incorrect diagnoses. It is a concrete provenance issue that needs explanation.',
            '', 'The lower HC amplitude also appears in the original128 BrainLat channels using a fixed56-second native-CAR/1–40Hz segment, **before interpolation**. Thus interpolation is not necessary to create this amplitude gap. These native-segment measurements are descriptive checks with a different fixed-window policy from the main feature experiment.',
            '', '| Nested SVM using only log10 median-channel RMS | Macro F1 | Balanced accuracy |','|---|---:|---:|']
        for g in ('OpenNeuro','BrainLat','Pooled'):
            m=amplitude[g]['mean'];lines.append(f'|{g}|{100*m["macro_f1"]:.1f}%|{100*m["balanced_accuracy"]:.1f}%|')
        lines+=['','The single-amplitude feature is almost sufficient for BrainLat diagnosis but does not transfer into a general cohort-independent disease rule. This can reflect biology, cohort composition, gain or prior processing; these data alone do not identify the cause. **Do not multiply HC signals by a diagnosis-specific factor:** that would require the answer at preprocessing time and distort the evaluation.',
            '', 'If different processing batches caused the amplitude difference, their perfect alignment with BrainLat labels prevents statistical correction from reliably separating batch effects from diagnosis. Obtain acquisition gain/unit information, original preprocessing scripts/settings and ideally raw recordings that can be processed consistently across diagnostic groups. Site z-scoring or CORAL cannot establish that this issue has been resolved.']
    lines+=['','## Interpretation and limits','',
        'Matched processing substantially reduces some acquisition differences, but compatibility and prediction performance are separate outcomes. Removing global amplitude or shared activity can also remove disease information. Do not export a single cohort-normalized feature matrix and then cross-validate it: learned site correction must remain inside the training pipeline.',
        '', 'The bootstrap intervals resample participants together with all their repeated held-out predictions. They are conditional on these fitted models and do not include all variability from retraining or recruitment. Repeated folds are not independent new participants. These cohorts have already informed earlier project experiments, so results remain exploratory rather than untouched external validation.',
        '', 'A defensible next analysis should predefine the correction and evaluate on new participants/sites. Keeping source/site metadata, matched referencing/filtering, separate cohort results and the uncorrected comparator is essential. Additional aggressive cleaning or channel deletion is not justified by cohort differences alone.',
        '', '## Files','',
        '- `signal_audit.json`: cohort summaries, age, within-diagnosis tests and channel/band effect sizes/FDR.',
        '- `prepared/parts/*.json`: per-record source hashes, signal metrics, accepted/rejected-window counts.',
        '- `evaluation/protocol.json` and `source_snapshot/`: frozen model protocol/code.',
        '- `evaluation/fold_*.json`: exact participant indices, inner searches and held-out predictions.',
        '- `evaluation/summary.json`, `predictions.npz`, `transfer.json`: machine-readable results.',
        '- `followup/native_metadata.json` and `amplitude_probe.json`: exploratory native-montage/provenance and nested one-feature checks.',
        '- `figures/*.png`: Matplotlib figures, no PDF.','']
    figure_names=['signal_differences','spectral_comparison','channel_band_differences','compatibility_vs_diagnosis','confusion_matrices','cross_dataset_transfer']
    if native:figure_names.append('amplitude_and_version_warning')
    for name in figure_names:
        lines.append(f'![{name}]({(FIG/(name+".png")).as_posix()})\n')
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Signal/model report and{len(figure_names)} Matplotlib figures saved.',flush=True)


if __name__=='__main__':main()
