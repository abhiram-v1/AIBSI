"""Build a Markdown research record from saved artifacts; never fit a model."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiment_search_records'
MAIN=ROOT/'EXPERIMENT_SEARCH_DETAILS.md'
SCOPES={
 '01_binary_logmel':('174-person binary log-mel search',ROOT/'neural_comparison/binary_logmel_search/outputs/nested'),
 '02_threeclass_spectral_neural':('Three-class spectral SVM, CNN, EEGPT and fusion',ROOT/'neural_comparison/advanced_clean_spectral/outputs'),
 '03_graph_ntf':('Full graph-NTF and matched tensor controls',ROOT/'neural_comparison/graph_ntf_svm/outputs/full'),
 '04_openneuro_fourier':('88-person Fourier SVM searches',ROOT/'neural_comparison/openneuro_fourier/outputs/evaluation'),
 '05_brainlat_native':('86-person native/interpolated montage ablation',ROOT/'neural_comparison/brainlat_native_ablation/outputs/matched'),
 '06_compatibility':('Cohort compatibility and amplitude investigations',ROOT/'neural_comparison/dataset_compatibility/outputs'),
 '07_literature_adaptations':('Literature method adaptations, full174 run',ROOT/'neural_comparison/paper_benchmark/outputs/full_174'),
 '08_source_transfer':('OpenNeuro-to-BrainLat transfer investigations',ROOT/'neural_comparison/source88_brainlat_transfer/outputs'),
 '09_earlier_tensor':('Earlier tensor and binary benchmark versions',ROOT/'tensor_pipeline/outputs'),
 '10_initial_stft':('Initial STFT neural/ML experiments',ROOT/'neural_comparison/stft_gpu_3class/outputs/run'),
}
IGNORE_PARTS={'prepared','cache','bases','models','parts','participant_parts','source_snapshot','validation_source_snapshot','synthetic_smoke','interpretation'}
IGNORE_FILES={'oof_predictions.json','predictions.json','retained_windows.json','attention.json','associations.json','severity_associations.json',
              'native_metadata.json','signal_audit.json','amplitude_literature_search_2026-11-05.json','svm_input_matrix.json'}
DROP={'train','test_indices','test_subjects','training_ids','train_ids','test_ids','fit_subjects','validation_subjects','training_subjects',
      'participant_ids','participant','participants','prediction','predictions','predicted','probabilities','truth','true','y','score_sum',
      'source_hashes','input_hashes','code_sha256','hashes','signature','matrix','selected_indices','selected_feature_names','test_truth',
      'test_predictions','scores_oof','source_oof_score','source_metadata_sha256','target_metadata_sha256'}
CHILD_LISTS={'candidates','validation_candidates','top_candidates','trials','inner_results','inner_provenance','history','threshold_scores',
             'development_folds','folds','records','calibration_folds','individual_runs','results','search_results','cv_results'}
METRIC_RE=re.compile(r'(^|_)(f1|accuracy|balanced|recall|precision|auc|score|loss|success|iterations|converged)(_|$)|^se_f1$|^balanced$|^ftd$')
PARAM_KEYS={'config','configuration','params','parameters','features','svm','chosen_config','selected','selected_on_validation','selection',
            'best','kernel','C','gamma','pca','rank','ranks','method','model','family','representation','normalization','site_adjustment',
            'scaler','feature_count','k','threshold','epoch','stage','seed','learning_rate','bag_size','epochs_max','max_epochs','patience',
            'extra_transformer_blocks','architecture','svm_weight','svm_temperature','eegpt_temperature','sampling_hz','segment_seconds',
            'stft_seconds','hop_seconds','mel_bins','summary','class_weight','weight','ftd_boost','channel_rank','spectral_rank','covariance','band_set'}
CONTAINERS={'metrics','test','validation','val','ensemble_test','ensemble_validation','inner_metrics','outer_metrics','summary','per_class','baseline','original_svm_test'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def link(path,label=None):
    path=Path(path).resolve();return f'[{label or path.name}]({path.as_posix()})'
def cell(value):
    if isinstance(value,float):text=repr(value)
    elif value is None:text='None'
    elif isinstance(value,(dict,list,tuple)):text=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    else:text=str(value)
    return text.replace('|',r'\|').replace('\r',' ').replace('\n','<br>')
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
            *['| '+' | '.join(cell(v) for v in row)+' |' for row in rows]]
def flatten(value,prefix='',exclude=()):
    result={}
    if isinstance(value,dict):
        for k,v in value.items():
            if k in DROP or k in exclude or k.startswith('source_') and 'hash' in k:continue
            p=f'{prefix}.{k}' if prefix else k
            if isinstance(v,dict):result.update(flatten(v,p,exclude))
            elif isinstance(v,list):
                if len(v)<=32 and all(not isinstance(z,(dict,list)) for z in v):result[p]=v
            elif isinstance(v,(str,int,float,bool)) or v is None:result[p]=v
    return result
def metric_flat(row):
    out={}
    for k,v in row.items():
        if k in CONTAINERS or k.endswith('_metrics'):
            for name,val in flatten(v,k,exclude=CHILD_LISTS).items():
                if METRIC_RE.search(name.split('.')[-1]) and isinstance(val,(int,float,list)):out[name]=val
        elif k in {'error','status','valid','invalid','fit_status'} and isinstance(v,(str,bool,int)):
            out[k]=v
        elif METRIC_RE.search(k) and k not in {'threshold_scores','score_sum'}:
            if k=='score' and isinstance(v,list):continue
            if isinstance(v,(int,float,bool,list)) and (not isinstance(v,list) or len(v)<=32 and all(isinstance(z,(int,float)) for z in v)):out[k]=v
    return out
def parameter_flat(row):
    out={}
    for k,v in row.items():
        if k in DROP or k in CHILD_LISTS or k in CONTAINERS or METRIC_RE.search(k):continue
        if k in PARAM_KEYS or k in {'candidate','repeat','fold','split','stage_epoch','ftd_multiplier','healthy_weight','terminal_rank','lam','coupling','kind',
                                    'n_estimators','max_depth','subsample','colsample_bytree','reg_alpha','reg_lambda','min_child_weight','selection_k','selection'}:
            if isinstance(v,dict):
                for name,val in flatten(v,k,exclude=CHILD_LISTS|CONTAINERS).items():
                    if not METRIC_RE.search(name.split('.')[-1]):out[name]=val
            elif isinstance(v,(str,int,float,bool)) or v is None:out[k]=v
            elif isinstance(v,list) and len(v)<=32 and all(not isinstance(z,(dict,list)) for z in v):out[k]=v
    return out
def extract(obj,pointer='$',context=None):
    context=context or {};out=[]
    if isinstance(obj,dict):
        params=parameter_flat(obj);scores=metric_flat(obj)
        if scores and params:
            out.append({'pointer':pointer,'parameters':{**context,**params},'scores':scores})
        child_context={**context}
        for key in ('candidate','fold','repeat','split','epoch','stage'):
            if key in obj and isinstance(obj[key],(str,int,float)):child_context[key]=obj[key]
        for key,value in obj.items():
            if key in DROP or key in {'per_class','signature','source_hashes','input_hashes','code_sha256','inner_splits','inner_provenance','calibration_splits'}:continue
            if isinstance(value,(dict,list)):out.extend(extract(value,f'{pointer}.{key}',child_context))
    elif isinstance(obj,list):
        for i,value in enumerate(obj):
            if isinstance(value,(dict,list)):out.extend(extract(value,f'{pointer}[{i}]',context))
    return out
def compact(mapping):return '; '.join(f'{k}={cell(v)}' for k,v in mapping.items())


def export_archives():
    OUT.mkdir(exist_ok=True);manifest=[];archives=[]
    for key,(title,base) in SCOPES.items():
        lines=[f'# {title}: saved parameter/result tables','',
            'Generated from existing JSON artifacts without training or tuning any model. Metrics retain their original field names and units: classification metrics usually use0–1; losses,scalar scores and optimizer diagnostics retain native units. Error flags and invalid-score sentinels are preserved and are not valid performance estimates. These rows include candidates, selected settings and training epochs, so the row count is **not** a count of unique configurations or independent fits.',
            '', 'Each JSON-path column identifies the exact source object. `inner_`, `cv_`, `val_`, `test.` and other namespaces must be interpreted separately. A missing held-out score is not replaced with a validation score. Duplicated selected/candidate records are retained as source records.',
            '', 'Older logs may contain partial runs or methods with different preprocessing and validation. Their presence here records what was saved; it does not establish completion, comparability, or clinical validity.','']
        total=0;files=0
        if not base.exists():continue
        for path in sorted(base.rglob('*.json')):
            if any(p in IGNORE_PARTS for p in path.relative_to(base).parts) or path.name in IGNORE_FILES:continue
            if any(w in path.name for w in ('artifact_manifest','input_fingerprint','delivery_manifest','preparation_report')):continue
            value=read(path);rows=extract(value)
            if not rows:continue
            files+=1;total+=len(rows)
            lines += [f'## {path.relative_to(base).as_posix()}','',f'Source: {link(path)}. SHA256: `{sha(path)}`. Saved configured/scored records: **{len(rows)}**.','']
            lines += table(['JSON path','Parameters / context','Saved results (original metric names)'],
                           [(r['pointer'],compact(r['parameters']),compact(r['scores'])) for r in rows])+['']
            manifest.append({'source':path.resolve().as_posix(),'sha256':sha(path),'archive':key+'.md','records':len(rows)})
        dest=OUT/(key+'.md');dest.write_text('\n'.join(lines),encoding='utf-8')
        archives.append((title,files,total,dest));print(f'{key}: {files} sources,{total} configured/scored records',flush=True)
    return archives,manifest


def literal_constants(path):
    out={}
    for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign):pairs=[(t,node.value) for t in node.targets]
        elif isinstance(node,ast.AnnAssign):pairs=[(node.target,node.value)]
        else:continue
        for target,value in pairs:
            if isinstance(target,ast.Name) and target.id.isupper():
                try:out[target.id]=ast.literal_eval(value)
                except (ValueError,TypeError):pass
    return out


def main_report(archives,manifest):
    nc=ROOT/'neural_comparison';adv=nc/'advanced_clean_spectral'
    inventory=read(nc/'final_experiment_inventory.json')
    binary=read(nc/'binary_logmel_search/outputs/nested/summary.json')
    selected=read(nc/'binary_logmel_search/outputs/nested/final/metadata.json')['selected']
    plan=read(nc/'binary_logmel_search/outputs/prepared/plan.json')
    candidates=read(nc/'binary_logmel_search/outputs/nested/final/search.json')['candidates']
    bselected=selected['config'];svm=read(adv/'outputs/results/site_normalized_metrics.json')
    eeg=read(adv/'outputs/deep_models/eegpt_attention/metrics.json');plus2=read(adv/'outputs/deep_models/eegpt_attention_plus2/metrics.json')
    blend=read(adv/'outputs/deep_models/eegpt_attention/svm_blend_metrics.json')
    extended=read(adv/'outputs/site_svm_extended_tuning/metrics.json');tt=read(adv/'outputs/tensor_train_svm/metrics.json')
    tucker=read(adv/'outputs/prepool_tucker_svm/metrics.json');riemann=read(adv/'outputs/riemannian_svm/metrics.json')
    frozen=read(adv/'outputs/frozen_eegpt_cv/frozen_eegpt_cv.json')
    graph=read(nc/'graph_ntf_svm/outputs/full/protocol.json');gfinal=read(nc/'graph_ntf_svm/outputs/full/final_gntf/metadata.json')
    lines=['# EEG experiment and parameter-search record','',
        '**Recorded:5 November2026.** This is a record of completed searches, saved configurations and their recorded results. It does not run new experiments or claim that all possible hyperparameters were exhausted.',
        '', '## What the scores mean','',
        '- **Inner CV / development CV / validation:** used to choose settings. These scores are not held-out test scores.',
        '- **Outer participant CV / held-out test:** used to assess a selected procedure. Different outer folds can choose different settings.',
        '- **Final refit:** selected using all available source participants for inference; it supplies no new independent test result.',
        '- Binary HC versus AD+FTD and three-class AD/FTD/HC are different tasks. OpenNeuro88, BrainLat86 and pooled174 must remain separate.',
        '- All main-summary metrics below are percentages. Detailed appendices preserve the original0–1 values and exact metric names.',
        '- Macro recall equals balanced accuracy when the same classes are included. A paper’s positive-class F1 must not be relabeled macro F1.',
        '- The27-person test split was examined over many model iterations. Its scores are exploratory. Repeated CV reuses participants; it does not create additional independent people.',
        '', '## Main binary log-mel search:174 participants','',
        f'**{len(plan["candidates"])} candidate configurations, {len(plan["representations"])} signal representations.** Fixed seeded finite random search plus matched anchors, with3 repeats ×5 outer folds and3 inner folds. Valid representation combinations require enough STFT frames. Every learned normalization, scaler, PCA, SVM and threshold is fitted/selected inside its applicable training fold.',
        '', 'This was a broad bounded search, not an exhaustive Cartesian search or an exhaustive optimization of CNN/EEGPT architecture.', '']
    names={'sampling_hz':'Sampling frequency (Hz)','segment_seconds':'Main window (s)','stft_seconds':'STFT window (s)','hop_seconds':'STFT hop (s)',
           'mel_bins':'Mel bins','summary':'Summary features','site_adjustment':'Site adjustment','scaler':'Scaler','pca':'PCA dimensions','kernel':'SVM kernel',
           'gamma':'RBF gamma','healthy_weight':'HC/class-weight option'}
    rows=[]
    for key,label in names.items():
        vals=sorted({cell(c[key]) for c in plan['candidates']});winner=bselected[key]
        if key=='gamma':winner='Not used by selected linear SVM'
        rows.append((label,', '.join(vals),winner))
    rows += [('SVM C','Sampled approximately0.001–100, plus fixed anchors',bselected['C']),('Decision-score threshold','−0.5,0,+0.5',bselected['threshold'])]
    lines+=table(['Parameter','Explored values','Final source-CV-selected refit'],rows)
    lines+=['','`four_stats` means mean, SD, median and IQR. `mean_sd` means mean and SD. The final model uses19×32×4=2432 participant features. Its final source-CV F1 is'+f' **{100*selected["inner_macro_f1"]:.2f}%**, a selection score, not the result to report as external performance.',
        '', 'The nested evaluation of the entire search procedure gave69.85% macro F1,70.29% balanced accuracy and72.03% accuracy. Matched128Hz and256Hz controls both gave69.31% macro F1 and70.18% balanced accuracy; their522 repeated held-out decisions were identical.',
        '', f'Sources: {link(nc/"binary_logmel_search/outputs/nested/REPORT.md")}, {link(nc/"binary_logmel_search/outputs/nested/final/search.json")}.',
        '', '### Every final-refit candidate and its validation score','',
        f'The256-row table and all three thresholds per candidate are in {link(OUT/"binary_logmel_all_candidates.md","the complete binary candidate table")}. All15 outer-fold search logs are also tabulated in archive01 below. Candidate IDs are stable across the binary search logs.',
        '', '## Three-class SVM, EEGPT and their probability blend','',
        'Pooled174-person cohort;121 training,26 validation and27 test participants. All rows in this table use that same historical fixed split. They are separate from the later frozen-EEGPT repeated-CV experiment.', '']
    rows=[]
    for name,m in [('Site-normalized log-mel SVM',svm['test']),('Fine-tuned EEGPT + attention',eeg['test']),('EEGPT + two added blocks',plus2['test']),('SVM + EEGPT probability blend',blend['test'])]:
        rows.append((name,*[f'{100*m[k]:.2f}' for k in ('accuracy','macro_f1','balanced_accuracy')]))
    lines+=table(['Method','Test accuracy (%)','Test macro F1 (%)','Test balanced accuracy (%)'],rows)
    lines+=['','### SVM search and selected configurations','',
        'The original site-normalized comparison tested two preprocessing representations, direct and hierarchical classifiers, PCA None/20/40/60, linear/RBF kernels, and C0.01/0.1/1/10. Gamma was scale; class weighting was balanced. It selected moderate log-mel,20 PCA components,linear C0.1. Signal settings were128Hz,8-second windows,2-second STFT,0.5-second hop,24 mel bands, with per-person mean/SD features.',
        '', 'The extended search used the following declared space (source constants read from the saved script):','']
    const=literal_constants(adv/'tune_site_svm_extended.py')
    lines+=table(['Parameter','Values'],[(k,const[k]) for k in ('PCA_COMPONENTS','C_VALUES','RBF_GAMMAS','FTD_BOOSTS','CV_SEEDS')])
    lines+=['',f'Extended candidates: **{extended["candidate_count"]}**, evaluated using2×5 development folds. Selected parameters: `{cell(extended["selected"])}`.',
        '', 'The development-selected extension scored66.67% test accuracy,61.01% macro F1 and61.41% balanced accuracy. It did not improve the original SVM’s test F1. Only the top50 candidate summaries were retained; scores for the other1518 configurations are not recoverable from the saved artifact without rerunning.',
        '', f'Sources: {link(adv/"outputs/results/site_normalized_metrics.json")}, {link(adv/"outputs/site_svm_extended_tuning/metrics.json")}.',
        '', '### EEGPT settings:fixed choices and limited architecture comparisons','',
        'The fine-tuned EEGPT run did not perform a full learning-rate/batch-size/dropout/depth search. The values below are documented settings used, not proven globally optimal values. The best epoch was selected on validation macro F1.','']
    eegsettings=[('Input','19 channels,4 seconds,256Hz; raw EEG'),('Checkpoint','Official58-channel4-second large4E EEGPT encoder'),
        ('Head warmup','6 epochs; frozen pretrained encoder'),('Fine-tuning','Last2 pretrained blocks plus normalization/channel-embedding/summary-token parameters'),
        ('Encoder learning rate',1e-5),('Warmup head learning rate',3e-4),('Fine-tuning head learning rate',1.5e-4),
        ('Optimizer / weight decay','AdamW /0.003'),('Scheduler','Cosine annealing; minimum LR2e-6'),('Bag size / training batch','8 windows per person /4 people'),
        ('Early-stopping patience','8; training stopped at epoch17'),('Selected checkpoint',f'epoch{eeg["best_epoch"]}'),
        ('Loss / gradient clip','Cross entropy, label smoothing0.06 / norm1.0'),('Projection dropout',.25),('Attention dropout',.10),
        ('Classifier dropout','0.35 and0.20'),('Training seed',eeg['seed']),('Added-depth ablation','0 versus2 new512-dimensional8-head blocks; extra-block dropout0.10')]
    lines+=table(['Setting','Recorded value'],eegsettings)
    lines+=['','The two-added-block model matched the original validation F1 but had worse held-out performance. This is a limited depth ablation, not a sweep over arbitrary transformer architectures.',
        '', f'Sources: {link(adv/"outputs/deep_models/eegpt_attention/metrics.json")}, {link(adv/"train_eegpt_attention.py")}, {link(adv/"RUN_SUMMARY.md")}. Some settings were reconstructed from code/README; the original metrics JSON did not persist every CLI argument. The saved epoch history confirms6 warmup and11 fine-tuning epochs.',
        '', '### SVM + EEGPT blend search','']
    lines+=table(['Parameter','Searched values','Selected'],[
        ('SVM temperature',[.5,.75,1.,1.5,2.],blend['selection']['svm_temperature']),
        ('EEGPT temperature',[.75,1.,1.25,1.5],blend['selection']['eegpt_temperature']),
        ('SVM probability weight','0–1 in0.05 increments',blend['selection']['svm_weight'])])
    lines+=['',f'**{len(blend["validation_candidates"])} saved validation combinations.** The chosen blend used50% SVM and50% EEGPT. Its validation macro F1 was71.00%; test macro F1 was67.82%. Every retained blend combination is in archive02.',
        '', '## Later repeated-CV frozen-EEGPT comparison','',
        'Pooled174 people;5 repeats ×5 outer folds;4 inner folds. The EEGPT encoder stays frozen and provides label-free embeddings. Each training fold fits its own normalization, PCA and linear SVM. This is a different procedure from fine-tuning the neural attention head above.', '']
    lines+=table(['Searched parameter','Values'],[(k,v) for k,v in frozen['protocol']['grid'].items()])
    lines+=['']
    rows=[]
    for method,s in frozen['summary'].items():
        rows.append((method,*[f'{100*s[k]["mean"]:.2f}' for k in ('accuracy','macro_f1','macro_recall')],f'{100*s["pooled_macro_f1_per_repeat"]["mean"]:.2f}'))
    lines+=table(['Representation/classifier','Mean fold accuracy (%)','Mean fold macro F1 (%)','Mean fold balanced accuracy (%)','Mean per-repeat pooled F1 (%)'],rows)
    lines+=['','Early fusion concatenates separately normalized/reduced spectral and2048-dimensional EEGPT blocks before SVM training. Late fusion uses a fixed equal-weight combination of classifier outputs. The scores are not interchangeable with the27-person fine-tuned probability blend.',
        '', 'Only selected settings per outer fold were retained here, not every inner candidate’s score. The90-setting single-block and60-setting fused searches cannot be reconstructed candidate-by-candidate from this JSON. All saved selections, metrics and marginal selection counts are recorded in archive02.',
        '', f'Source: {link(adv/"outputs/frozen_eegpt_cv/frozen_eegpt_cv.json")}.',
        '', '## Tensor searches','']
    tensor_blend = adv / 'outputs/tensor_eegpt_blend_2026-11-05'
    if (tensor_blend / 'metrics.json').exists():
        fresh = read(tensor_blend / 'metrics.json')
        historical = read(tensor_blend / 'historical_tt_control/metrics.json')
        lines += ['### Newly verified TT-SVM + EEGPT ensemble', '',
                  'EEGPT was reused without retraining. The fresh TT/SVM search uses five inner folds within121 training participants; its blend is tuned on26 validation participants. A separate follow-up control fixes the previously selected TT/SVM settings. Both refit the tensor branch on147 development participants and use the historical27-person test set. The control was added after the fresh run; both results are retained.', '']
        lines += table(['Model','Test accuracy (%)','Test macro F1 (%)','Test balanced accuracy (%)'],
                       [(label, *[f'{100*m[k]:.2f}' for k in ('accuracy','macro_f1','balanced_accuracy')])
                        for label,m in [('No-tensor SVM + EEGPT',fresh['test']['Spectral SVM + EEGPT']),
                                        ('Historical TT-SVM alone',historical['test']['Historical TT-SVM']),
                                        ('Historical TT-SVM + EEGPT',historical['test']['Historical TT-SVM + EEGPT tuned']),
                                        ('Fresh training-CV-selected TT-SVM + EEGPT',fresh['test']['TT-SVM + EEGPT tuned'])]])
        lines += ['', f'All288 SVM candidates and both420-combination blend searches are retained in archive02. See {link(tensor_blend/"REPORT.md")} and {link(tensor_blend/"historical_tt_control/REPORT.md")}. These runs do not reproduce an approximately70% tensor ensemble. The no-tensor70.37% blend and the old standalone TT51.85% were reproduced exactly.','']
    joint = adv / 'outputs/tensor_eegpt_joint_search_2026-11-05'
    if (joint/'metrics.json').exists():
        result = read(joint/'metrics.json')
        lines += ['### Joint tensor/SVM/EEGPT-blend search', '',
                  'A new joint search selected the tensor/SVM for validation ensemble F1 instead of standalone SVM F1. It searched720 tensor/SVM configurations ×420 blend choices =302,400 joint comparisons, including RBF gamma scale/0.01/0.1/1. EEGPT stayed fixed. Training-only five-fold SVM CV was a tie-break; all joint selection used26 validation participants. The test set remains the previously examined27-person split.', '']
        lines += table(['Model','Test accuracy (%)','Test macro F1 (%)','Test balanced accuracy (%)'],
                       [(name,*[f'{100*m[k]:.2f}' for k in ('accuracy','macro_f1','balanced_accuracy')]) for name,m in result['test'].items()])
        lines += ['', f'Selected settings: `{cell(result["selected"])}`. Validation ensemble F1 was80.11%; test F1 was58.73%. The large validation/test gap limits interpretation of further search gains. Full report: {link(joint/"REPORT.md")}; all joint trials: {link(joint/"all_joint_candidates.jsonl.gz")}; per-parameter winners: {link(joint/"parameter_best_results.json")}.','']
    const=literal_constants(adv/'train_tensor_train_svm.py')
    lines+=table(['Tensor Train parameter','Searched values'],[(k,const[k]) for k in ('RANKS','CHANNEL_RANKS','SPECTRAL_RANKS','SVM_C','SVM_KERNELS')])
    lines+=['',f'TT candidates: **{tt["candidate_count"]}** on2×5 development folds. Selected `{cell(tt["selected"])}`. Test F1={100*tt["test"]["macro_f1"]:.2f}%, balanced accuracy={100*tt["test"]["balanced_accuracy"]:.2f}%, accuracy={100*tt["test"]["accuracy"]:.2f}%. Only the top20 of288 candidate summaries and rank summaries were retained.',
        '', '### Tucker decomposition before pooling','']
    const=literal_constants(adv/'train_prepool_tucker_svm.py')
    lines+=table(['Parameter','Searched values'],[(k,const[k]) for k in ('RANKS','SUMMARIES','NORMALIZATIONS','PCA_COMPONENTS','C_VALUES','KERNELS')])
    lines+=['',f'Tucker candidates: **{tucker["candidate_count"]}**. Selected `{cell(tucker["selected"])}`. Test F1={100*tucker["test"]["macro_f1"]:.2f}%, balanced accuracy={100*tucker["test"]["balanced_accuracy"]:.2f}%, accuracy={100*tucker["test"]["accuracy"]:.2f}%. Only top20 candidates were retained.',
        '', '### Graph-regularized nonnegative tensor factorization','',
        'Full174-person run:3 repeats ×5 outer folds and3 inner folds. Final refit uses a one-standard-error simplicity rule. Rank and graph strength are searched jointly with SVM settings; final-refit scores are inner selection results, not held-out estimates.','']
    grids=graph['rank_grid'];lines+=table(['Parameter','Explored / recorded value'],[
        ('Feature methods',graph['methods']),('Ranks',sorted({v['rank'] for v in grids})),('Graph penalties',sorted({v.get('lam',0) for v in grids})),
        ('Connectivity coupling',sorted({v.get('coupling',0) for v in grids})),('Optimizer maximum iterations / starts',f'{graph["maxiter"]} /{graph["starts"]}'),
        ('SVM grid',graph['svm_grid']),
        ('Final selected feature/SVM settings',gfinal['selected'])])
    lines+=['','The full search logs contain the exact SVM grids and candidate scores for direct,NTF,G-NTF,shuffled-graph andTT controls; all are tabulated in archive03. The final simpler G-NTF fit uses rank5,graph penalty0.01 andlinear C10. The maximal inner-CV candidate was rank12/penalty0.001/C3; it was not the final selected simpler model.',
        '', 'The full G-NTF outer macro F1 was41.1%, balanced accuracy47.2%, accuracy53.4%. Several tensor fits reached the iteration budget without optimizer convergence, and source preparation lacked explicit discontinuity rejection; these limit interpretation. Reconstruction, smooth maps or attractive factors do not establish improved classification or anatomical causality.',
        '', f'Source report: {link(nc/"graph_ntf_svm/outputs/full/REPORT.md")}.',
        '', '## Latest OpenNeuro Fourier search','',
        'OpenNeuro88 only. Fixed processing:256Hz,19-channel CAR,1–40Hz,8-second epochs; Welch4-second window/2-second hop/1024-point FFT,157 frequencies at0.25Hz spacing. **These signal parameters were fixed, not tuned in this experiment.**', '']
    four=read(nc/'openneuro_fourier/outputs/evaluation/protocol.json')
    lines+=table(['Searched parameter','Values'],[(k,four[k]) for k in ('feature_families','feature_counts','svm_grid','binary_score_thresholds','class_weights','pca')])
    lines+=['','3 repeats ×5 outer folds and3 inner folds. ANOVA feature ranking,feature count,scaling andSVM tuning are training-fold-only. Threshold tuning applies only to binary classification. All family/candidate/fold records are in archive04.','']
    rows=[]
    for task in ('binary','three_class'):
        summary=read(nc/f'openneuro_fourier/outputs/evaluation/{task}_summary.json')
        for arm,value in summary.items():rows.append((task,arm,*[f'{100*value["mean"][k]:.2f}' for k in ('macro_f1','balanced_accuracy','accuracy')]))
    lines+=table(['Task','Feature family / procedure','Outer macro F1 (%)','Outer balanced accuracy (%)','Outer accuracy (%)'],rows)
    lines+=['','The fixed relative-spectrum arm reached74.75% binary F1, while selecting the entire feature-family procedure inside inner folds gave71.94%. Reporting the highest outer result as if it were independently selected would overstate the evidence. No universal optimal configuration was selected across all folds.',
        '', '## BrainLat montage, harmonization and amplitude investigations','',
        'The86-person montage ablation compared native128,native19,legacy19 andaligned19 on the same participants and matched windows. It used256 classifier configurations per arm, with training-only site adjustment/scaling/PCA and source-selected thresholds. The biological question and electrode representation differ from the earlier174-person searches. All saved candidates and outer results are in archive05.',
        '', 'The compatibility audit fixed256Hz/8s/2s-STFT/0.5s-hop/24mel, compared six preprocessing/harmonization arms, and searched balanced linear/RBF SVM C0.01/0.1/1/10 with thresholds−0.5/0/0.5. PCA20 andCORAL shrinkage0.3 were fixed. Its source-only transfer comparisons do not fit target-cohort statistics.',
        '', 'The amplitude-only BrainLat pipeline used one log10 RMS feature with the same8 SVM configurations and3 thresholds. Its internal macro F1 was97.10%; the final BrainLat-only model selected linear C0.01,threshold0. Transferred to all88 OpenNeuro participants, it predicted every person as dementia:40.14% macro F1,50% balanced accuracy,0%HC recall. The amplitude/version concern remains unresolved; this is not proof of intentional manipulation. Archive06 contains all saved trials and selections.',
        '', '## What was not exhaustively searched','']
    lines+=table(['Area','Coverage and limitation'],[
        ('Binary signal/SVM search','256 sampled configurations with representation coverage; not the full Cartesian product.'),
        ('CNN / CNN-BiLSTM / attention','Architecture,seed,representation,training-length andcalibration comparisons are recorded; no exhaustive joint width/depth/LR/batch/dropout search.'),
        ('EEGPT fine-tuning','Documented fixed optimization settings and0-versus2-added-block ablation; no full nested fine-tuning hyperparameter search.'),
        ('Frozen EEGPT + SVM','Nested normalization/PCA/C/FTD-weight search; encoder architecture andweights fixed.'),
        ('Tensor optimization','Finite rank/regularization grids andfinite optimizer budgets; unconverged fits must remain flagged.'),
        ('Fourier signal settings','Main/Welch window,hop,sampling rate,frequency range fixed; only feature/classifier choices tuned.'),
        ('Partial logs','Some experiments retained only top candidates or selected settings; absent scores are not invented.'),
        ('Earlier binary tree work','Thev9 archive contains only8 saved folds; treat it as incomplete, not a completed15-fold benchmark.')])
    lines+=['','## Complete saved parameter/result appendices','',
        'The appendices provide every configured/scored record recoverable by the documented extractor from the included JSON scopes. They include candidate rows,epoch histories andselected settings. Their counts include repeated folds andduplicate selections; they are not counts of unique hyperparameter combinations. Original JSON links andhashes are retained for review.', '']
    lines+=table(['Experiment scope','Source JSON files with records','Configured/scored records','Markdown tables'],[(title,files,total,link(path,'Open tables')) for title,files,total,path in archives])
    lines+=['','The earlier tensor archive preserves historical experiments, including logistic regression,random forests,XGBoost,CP/NMF/Tucker/TT andbinary variants when their settings/scores were saved. Their own source protocols remain authoritative; cohort/task/evaluation differences prevent ranking all historical scores in one leaderboard.',
        '', '### Known score-retention gaps','']
    lines+=table(['Search','Declared candidate count','Candidate summaries retained','How to read the archive'],[
        ('Extended three-class SVM',1568,len(extended['top_candidates']),'Top50 only; selected andouter-test results also retained.'),
        ('Tensor Train + SVM',288,len(tt['top_candidates']),'Top20 only; terminal-rank summary also retained.'),
        ('Tucker before pooling',288,len(tucker['top_candidates']),'Top20 only; no remaining candidate scores in saved JSON.'),
        ('Riemannian SVM',1440,len(riemann['top_candidates']),'Top30 only.'),
        ('Frozen EEGPT nested SVM','90single-block /60fusion per inner search','Selected settings only','Full inner scores were not persisted.'),
        ('Binary log-mel',256,len(candidates),'All final-refit candidates andall15 outer inner-search tables retained.'),
        ('EEGPT probability blend',420,len(blend['validation_candidates']),'All validation combinations retained.'),
        ('OpenNeuro Fourier','Per-family/per-fold grid','All trials in completed30 outer-fold artifacts','Signal window parameters remain fixed.')])
    lines+=['','## Would more search help?','',
        'It could. These experiments do not establish an accuracy/F1 ceiling. A targeted future search could tune EEGPT learning rates,unfreezing depth,regularization andpooling using participant-level inner folds. Signal-processing choices could also be jointly tested in the Fourier pipeline. The current record does not claim that these searches were already completed.',
        '', 'Small score gains would need uncertainty estimates andan evaluation protocol chosen before looking at outer results. Reusing a repeatedly inspected split can reward selection noise. More search cannot establish whether the BrainLat amplitude difference is biological or caused by acquisition/preprocessing; that requires provenance ormatched raw-data processing.',
        '', 'This document is a record of the saved evidence, not a claim that the largest observed number is the final expected clinical performance.',
        '', f'Generation script: {link(Path(__file__))}. Source index andhashes: {link(OUT/"source_manifest.json")}. No model was fitted to create this record.','']
    # Light summary spacing; leave file links and the source-value appendices intact.
    prose=[]
    for line in lines:
        if line:
            pieces=re.split(r'(\[[^\]]+\]\([^)]+\))',line)
            for i in range(0,len(pieces),2):
                s=pieces[i]
                s=re.sub(r'\b(the|The|with|With|all|All|gave|Matched|their|Only|only|top|remaining|using|used|uses|and|or|for|from|epoch|rank|penalty|linear|approximately|PCA|C|Official|November|within|across|of|Pooled|on|in|scored|was|is|both|completed|other|Selected|selected|candidate|archive|Source|source)(?=\d)',r'\1 ',s)
                s=re.sub(r',(?=[A-Za-z0-9])',', ',s)
                s=re.sub(r'([:;])(?=[A-Za-z0-9])',r'\1 ',s)
                s=s.replace('original0–1','original 0–1').replace('OpenNeuro88','OpenNeuro (88)').replace('BrainLat86','BrainLat (86)').replace('pooled174','pooled (174)')
                for old,new in {'andCORAL':'and CORAL','andweights':'and weights','Thev9':'The v9','andouter':'and outer','andall':'and all','andSVM':'and SVM','andTT':'and TT','andlinear':'and linear','andan':'and an','ormatched':'or matched','andpooling':'and pooling','andhashes':'and hashes','classifiers,PCA':'classifiers, PCA'}.items():
                    s=s.replace(old,new)
                s=re.sub(r'(?<=\d)(Hz)\b',r' \1',s)
                s=re.sub(r'×(?=\d)','× ',s)
                pieces[i]=s
            line=''.join(pieces)
        prose.append(line)
    MAIN.write_text('\n'.join(prose),encoding='utf-8')
    # Human-readable full256-candidate table with every retained threshold result.
    cdoc=['# All 256 final-refit binary log-mel candidates','',
          'All scores below are source-cohort inner-CV selection scores on 174 participants. They are not outer-test or external-validation results. Final selected candidate 95. The full search procedure’s outer F1 is 69.85%. Percentages in this table are rounded to four decimal places; the general archive and source JSON preserve full precision.','']
    columns=['ID','Hz','Segment s','STFT s','Hop s','Mel','Statistics','Site','Scaler','PCA','Kernel','C','Gamma','HC weight','Chosen threshold','Inner F1 (%)','Inner BA (%)']
    rows=[];thresholds=[]
    for row in candidates:
        c=row['config'];rows.append([row['candidate'],*[c[k] for k in ('sampling_hz','segment_seconds','stft_seconds','hop_seconds','mel_bins','summary','site_adjustment','scaler','pca','kernel','C','gamma','healthy_weight','threshold')],f'{100*row["inner_macro_f1"]:.4f}',f'{100*row["inner_balanced_accuracy"]:.4f}'])
        for t in row['threshold_scores']:thresholds.append((row['candidate'],t['threshold'],f'{100*t["macro_f1"]:.4f}',f'{100*t["balanced_accuracy"]:.4f}'))
    cdoc+=table(columns,rows)+['','## Every threshold score','']+table(['Candidate','Threshold','Inner macro F1 (%)','Inner BA (%)'],thresholds)
    cdoc+=['',f'Source: {link(nc/"binary_logmel_search/outputs/nested/final/search.json")}.','']
    (OUT/'binary_logmel_all_candidates.md').write_text('\n'.join(cdoc),encoding='utf-8')
    return len(candidates),len(thresholds)


def main():
    archives,manifest=export_archives()
    # Write manifest before main document so its source link resolves immediately.
    (OUT/'source_manifest.json').write_text(json.dumps({'generator_sha256':sha(__file__),'sources':manifest},indent=2),encoding='utf-8')
    candidates,thresholds=main_report(archives,manifest)
    # Artifact checks: all emitted records have a source, every link resolves, tables have matching cells.
    assert candidates==256 and thresholds==768
    for record in manifest:assert Path(record['source']).exists() and sha(record['source'])==record['sha256']
    files=[MAIN,*OUT.glob('*.md')];links=0;tables=0
    for path in files:
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:')):continue
            assert Path(target).exists(),(path,target);links+=1
        expected=None
        for line in text.splitlines():
            if line.startswith('|'):
                cells=len(re.split(r'(?<!\\)\|',line))-2
                if expected is None:expected=cells;tables+=1
                assert cells==expected,(path,cells,expected,line[:200])
            else:expected=None
    verification={'passed':True,'markdown_files':len(files),'tables':tables,'resolved_file_links':links,
                  'source_files':len(manifest),'configured_scored_records':sum(v['records'] for v in manifest),
                  'binary_final_candidates':candidates,'binary_final_threshold_rows':thresholds,
                  'checks':['all generated file links resolve','every Markdown table row matches its header width',
                            'all archived records linked to SHA256-verified source JSON','all256 binary final candidates and768 threshold scores included'],
                  'no_models_trained':True,'limitations':'Configured/scored records include repeated/selected/epoch observations; they are not unique configuration counts.'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
