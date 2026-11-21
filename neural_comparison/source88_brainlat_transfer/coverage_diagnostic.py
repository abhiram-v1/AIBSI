"""Flagged coverage extension; models and strict-QC endpoint remain frozen."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from transfer_data import ROOT,HERE,windows_from_recording,feature_banks
from run import V6,NAMES,dump,sha,charts


def main():
    out=HERE/'outputs';original=json.loads((out/'summary.json').read_text())
    audit=json.loads((out/'target_audit.json').read_text());people=json.loads((out/'target_manifest.json').read_text())
    frozen=json.loads((out/'source_frozen.json').read_text());arrays=dict(np.load(out/'target_features.npz'))
    mapping=np.load(ROOT/'neural_comparison/brainlat_native_ablation/outputs/prepared/operators.npz')['legacy19']
    additions=[];parts={k:[] for k in arrays}
    for person in audit:
        if person['status']!='unscorable':continue
        raw,car,details=windows_from_recording(person,mapping,minimum_sd_uv=.1)
        features=feature_banks(raw,car)
        if not all(np.isfinite(v).all() for v in features.values()):raise ValueError('Non-finite diagnostic features')
        additions.append({**person,'status':'flagged_low_amplitude','diagnostic_audit':details})
        for k in parts:parts[k].append(features[k])
        print('Flagged prediction features:',person['subject_id'],flush=True)
    combined=people+additions
    x={k:np.vstack([arrays[k],np.stack(parts[k])]) if additions else arrays[k] for k in arrays}
    y=np.array([int(p['label']!='HC') for p in combined]);results={};predictions={}
    for name in NAMES:
        path=out/f'{name}.joblib'
        if sha(path)!=frozen['models'][name]['model_sha256']:raise ValueError('Source model changed')
        model=joblib.load(path);matrix=x[model.config['features']];pred=model.predict(matrix);p=model.predict_proba(matrix)
        result=V6.metrics(y,pred);result['source_config']=model.config;results[name]=result
        rows=[]
        for i,(person,label,target,prob) in enumerate(zip(combined,pred,y,p)):
            rows.append({'participant':person['subject_id'],'site':person['site'],'original_label':person['label'],
                'true_binary':int(target),'predicted_binary':int(label),'predicted_class':'Dementia' if label else 'HC',
                'probabilities':{'HC':float(prob[0]),'Dementia':float(prob[1])},
                'quality_status':'original_source_policy' if i<len(people) else 'below_original_amplitude_floor'})
        # The previously scored subset must remain exactly unchanged.
        old=json.loads((out/'predictions.json').read_text())[name]
        assert [r['predicted_binary'] for r in rows[:len(people)]]==[r['predicted_binary'] for r in old]
        predictions[name]=rows
    summary={'source_participants':88,'target_headers':len(audit),'target_scored':len(combined),
             'models':results,'readable_class_counts':{label:sum(p['label']==label for p in combined) for label in ['AD','FTD','HC']},
             'quality_flagged':len(additions),'missing_signal':sum(p['status']=='missing_signal_file' for p in audit),
             'note':'Coverage diagnostic only: source-policy76 plus10 low-amplitude controls using pre-existing0.1uV minimum; frozen models, no target fitting or selection. Strict-QC results remain the primary endpoint.',
             'always_dementia':V6.metrics(y,np.ones(len(y),int))}
    dest=out/'all_readable_diagnostic';dest.mkdir(exist_ok=True)
    dump(dest/'summary.json',summary);dump(dest/'predictions.json',predictions)
    dump(dest/'flagged_recordings.json',additions);np.savez_compressed(dest/'features.npz',**x)
    dump(dest/'protocol.json',{'original_source_frozen_sha256':sha(out/'source_frozen.json'),
        'strict_summary_sha256':sha(out/'summary.json'),'diagnostic_code_sha256':sha(__file__),
        'extraction_code_sha256':sha(HERE/'transfer_data.py'),'model_hashes_unchanged':True,
        'rule':'Retain original source-policy windows for76 scorable cases; only previously unscorable files use0.1uV floor instead of1uV. This is a flagged post-primary coverage diagnostic, not a newly tuned performance result.'})
    charts(dest,summary,predictions,y)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
