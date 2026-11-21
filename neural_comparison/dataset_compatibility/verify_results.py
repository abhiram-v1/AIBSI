"""Check output alignment, preprocessing provenance, CV partitions and saved metrics."""
import json
import numpy as np
from audit_data import HERE,dump,sha
from compat_model import ARMS,metrics


def main():
    out=HERE/'outputs';prep=out/'prepared';ev=out/'evaluation'
    ps=json.loads((prep/'plan.json').read_text())['participants'];data=dict(np.load(prep/'features.npz'))
    assert len(ps)==174 and len({p['subject_id'] for p in ps})==174
    for arm in ('raw','common','gain','relative'):
        assert data[arm].shape==(174,912) and np.isfinite(data[arm]).all()
    protocol=json.loads((ev/'protocol.json').read_text())
    assert sha(prep/'features.npz')==protocol['features_sha256']
    assert sha(prep/'plan.json')==protocol['plan_sha256']
    for file,h in protocol['code_sha256'].items():
        assert sha(HERE/file)==sha(ev/'source_snapshot'/file)==h
    signature=json.loads((prep/'parts/000.json').read_text())['signature']
    for i,p in enumerate(ps):
        meta=json.loads((prep/'parts'/f'{i:03d}.json').read_text())
        assert meta['participant']==p
        assert meta['signature']['data_code_sha256']==sha(HERE/'audit_data.py')
        assert meta['signature']['operators_sha256']==signature['operators_sha256']
        assert meta['windows']['accepted_windows']>=3
        assert np.isfinite(list(meta['raw_metrics'].values())).all()
        assert np.isfinite(list(meta['common_metrics'].values())).all()
    pred=dict(np.load(ev/'predictions.npz'));summary=json.loads((ev/'summary.json').read_text())
    coverage=np.zeros((3,174),int)
    for file in ev.glob('fold_*.json'):
        record=json.loads(file.read_text());tr=np.array(record['train']);te=np.array(record['test']);r=record['repeat']
        assert len(set(tr)&set(te))==0 and sorted([*tr,*te])==list(range(174))
        coverage[r,te]+=1
        for arm in ARMS:
            entry=record['arms'][arm];p=np.array(entry['prediction'])
            assert np.array_equal(p,np.array(entry['score'])>=entry['best']['config']['threshold'])
            assert np.array_equal(p,pred[f'diagnosis_{arm}'][r,te])
        chosen=record['selected_arm']
        assert np.array_equal(record['arms'][chosen]['prediction'],pred['diagnosis_selected'][r,te])
    assert np.all(coverage==1)
    for arm in (*ARMS,'selected'):
        for r in range(3):assert metrics(pred['y'],pred[f'diagnosis_{arm}'][r])==summary[arm]['per_repeat'][r]
    for record in json.loads((ev/'transfer.json').read_text()):
        tr=np.array(record['training']);te=np.array(record['test']);assert not set(tr)&set(te)
        assert len(set(pred['domain'][tr]))==1 and len(set(pred['domain'][te]))==1
        assert pred['domain'][tr[0]]!=pred['domain'][te[0]]
        for entry in record['arms'].values():
            assert metrics(pred['y'][te],entry['prediction'])==entry['metrics']
    native=json.loads((out/'followup/native_metadata.json').read_text())
    assert [m['subject_id'] for m in native]==[p['subject_id'] for p in ps]
    for m in native:
        if m['source']=='brainlat':assert m['header_eeglab_version']==('2022.1' if m['label']=='HC' else '2020.0')
    for m in json.loads((out/'followup/amplitude_probe.json').read_text()).values():
        coverage={}
        for rec in m['records']:
            assert not set(rec['training'])&set(rec['test'])
            for i in rec['test']:coverage[(rec['repeat'],i)]=coverage.get((rec['repeat'],i),0)+1
        assert len(coverage)==3*len(m['indices']) and set(coverage.values())=={1}
    dump(out/'verification.json',{'passed':True,'participants':174,'features_per_person':912,
        'main_outer_folds':15,'main_diagnosis_pipelines_per_fold':6,'repeats':3,
        'checks':['feature dimensions/finiteness','fixed participant IDs and metadata alignment',
                  'code/snapshot/feature/plan hashes','one held-out prediction per person per repeat',
                  'disjoint train/test folds','saved metrics and score-threshold consistency',
                  'source-only transfer partitions','native metadata warning confirmed for all86 BrainLat records',
                  'exploratory one-feature probe train/test coverage']})
    print('Output verification passed.',flush=True)


if __name__=='__main__':main()
