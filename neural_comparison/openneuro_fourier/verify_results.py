"""Validate person-level matrices, frozen code, CV coverage and saved predictions."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_data import HERE,AUDIT,DESCRIPTORS,RELATIVE_INDICES,dump,sha
from fourier_model import ARMS,metrics,train_predict


def main():
    out=HERE/'outputs';prep=out/'prepared';ev=out/'evaluation'
    ps=json.loads((prep/'plan.json').read_text())['participants'];a=dict(np.load(prep/'features.npz'))
    names=json.loads((prep/'feature_names.json').read_text());p=json.loads((ev/'protocol.json').read_text())
    assert len(ps)==88 and len({v['subject_id'] for v in ps})==88
    assert all(v['source']=='ds004504' for v in ps)
    assert a['psd'].shape==(88,19,157) and a['summary'].shape==(88,361) and a['relative_summary'].shape==(88,247)
    np.testing.assert_array_equal(a['frequency'],np.arange(1,40.01,.25))
    for arm in ARMS:assert a[arm].shape==(88,len(names[arm])) and np.isfinite(a[arm]).all()
    np.testing.assert_allclose(a['relative_psd'].sum(2)*.25,1,atol=1e-12)
    np.testing.assert_allclose(a['summary_tensor'][:,:,5:10].sum(2),1,atol=1e-12)
    assert sha(prep/'features.npz')==p['features_sha256'] and sha(prep/'plan.json')==p['plan_sha256']
    for file,h in p['code_sha256'].items():assert sha(HERE/file)==sha(ev/'source_snapshot'/file)==h
    dependency=sha(AUDIT/'compat_model.py')
    prior_protocol=json.loads((AUDIT/'outputs/evaluation/protocol.json').read_text())
    assert dependency==prior_protocol['code_sha256']['compat_model.py']
    for i,person in enumerate(ps):
        meta=json.loads((prep/'parts'/f'{i:03d}.json').read_text());assert meta['participant']==person
        assert meta['signature']['code_sha256']==sha(HERE/'spectral_data.py')
        assert meta['signature']['audit_data_sha256']==sha(AUDIT/'audit_data.py')
        old=json.loads((AUDIT/'outputs/prepared/parts'/f'{i:03d}.json').read_text())
        np.testing.assert_allclose(10**a['amplitude'][i,0],old['common_metrics']['median_channel_rms_uv'],rtol=1e-12)
        assert len(meta['starts_samples'])==meta['windows']['accepted_windows']>=3
    reproduced=[]
    for task in ('binary','three_class'):
        pred=dict(np.load(ev/f'{task}_predictions.npz'));s=json.loads((ev/f'{task}_summary.json').read_text())
        classes=2 if task=='binary' else 3;coverage=np.zeros((3,88),int)
        for file in ev.glob(f'{task}_[0-9]_[0-9].json'):
            rec=json.loads(file.read_text());tr=np.array(rec['train']);te=np.array(rec['test']);r=rec['repeat']
            assert not set(tr)&set(te) and sorted([*tr,*te])==list(range(88));coverage[r,te]+=1
            for arm in ARMS:
                entry=rec['arms'][arm];np.testing.assert_array_equal(entry['prediction'],pred[arm][r,te])
                assert entry['selected_feature_names']==[names[arm][i] for i in entry['selected_indices']]
                if classes==2:np.testing.assert_array_equal(np.array(entry['score'])>=entry['best']['config']['threshold'],entry['prediction'])
            arm=rec['selected_arm'];np.testing.assert_array_equal(rec['arms'][arm]['prediction'],pred['selected'][r,te])
            if r==0 and rec['fold']==0:
                entry=rec['arms'][arm];pp,ss,ii=train_predict(a[arm][tr],pred['y'][tr],a[arm][te],entry['best']['config'],classes)
                np.testing.assert_array_equal(pp,entry['prediction']);np.testing.assert_allclose(ss,entry['score'],rtol=1e-10,atol=1e-10)
                assert ii==entry['selected_indices'];reproduced.append(f'{task}:repeat0/fold0/{arm}')
        assert np.all(coverage==1)
        for arm in (*ARMS,'selected'):
            runs=[metrics(pred['y'],row,classes) for row in pred[arm]];assert runs==s[arm]['per_repeat']
            for key in ('macro_f1','balanced_accuracy','accuracy'):assert np.isclose(np.mean([v[key] for v in runs]),s[arm]['mean'][key])
        if task=='binary':
            previous=json.loads((AUDIT/'outputs/followup/amplitude_probe.json').read_text())['OpenNeuro']
            assert s['amplitude']['mean']=={k:previous['mean'][k] for k in ('macro_f1','balanced_accuracy','accuracy')}
    association=json.loads((out/'associations.json').read_text())
    for family,contrasts in association['families'].items():
        for values in contrasts.values():
            for item in values:
                for key in ('mann_whitney_p','fdr_q','adjusted_p','adjusted_fdr_q'):assert 0<=item[key]<=1
                assert item['group0_n']+item['group1_n']<=88
    assert len(json.loads((out/'severity_associations.json').read_text())['tests'])==380
    dump(out/'verification.json',{'passed':True,'participants':88,'outer_folds_per_task':15,'tasks':2,
        'frozen_shared_svm_dependency_sha256':dependency,'reproduced_heldout_models':reproduced,
        'checks':['feature dimensions,units and relative-power normalization','unique subject IDs and OpenNeuro-only cohort',
                  'feature/protocol/snapshot hashes','same RMS measurements and binary baseline as previous audit',
                  'complete disjoint participant CV partitions','exact saved metric reproduction',
                  'feature-name/index alignment','independent refit of one selected held-out model per task',
                  'valid descriptive p/q values and severity sample size']})
    print('Fourier result verification passed.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
