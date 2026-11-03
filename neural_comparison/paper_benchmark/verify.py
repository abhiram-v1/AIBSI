"""Audit all fold memberships and replay the saved outer-test predictions."""
from pathlib import Path
import json

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from prepare import OUT as DATA, HERE, ROOT, sha, write_json
from models import DICENet, MicrostateNFNet, microstate_sequences
from run import METHODS, metric, evaluate_windows, report


def main():
    out=HERE/'outputs/full_174';protocol=json.loads((out/'protocol.json').read_text())
    for name,digest in protocol['code_hashes'].items():
        assert sha(HERE/name)==digest, f'Code changed after freezing: {name}'
    assert sha(DATA/'manifest.json')==protocol['data_manifest_sha256']
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in manifest['array_hashes'].items():assert sha(DATA/name)==digest
    for name,digest in protocol['native_and_micro_input_hashes'].items():assert sha(DATA/name)==digest
    d=np.load(DATA/'subjects.npz');y=d['y']
    assert len(y)==174 and np.bincount(y).tolist()==[71,42,61]
    assert manifest['duplicate_cross_participant_windows']==0
    checked=[]
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for fold in range(5):
            for method in METHODS:
                folder=out/f'fold_{fold:02d}'/method;r=json.loads((folder/'result.json').read_text())
                train=np.array(r['train_indices']);test=np.array(r['test_indices'])
                assert train.tolist()==protocol['outer_splits'][fold]['train']
                assert test.tolist()==protocol['outer_splits'][fold]['test']
                assert not set(train)&set(test)
                if method==METHODS[0]:
                    saved=joblib.load(folder/'model.joblib')
                    assert saved['train_indices'].tolist()==train.tolist()
                    p=saved['classifier'].predict_proba(saved['representation'].transform(d['psd_db'][test]))
                    pred=p.argmax(1)
                    assert np.allclose(p,r['details']['probabilities'],atol=1e-10)
                elif method==METHODS[1]:
                    saved=joblib.load(folder/'model.joblib')
                    assert saved['train_indices'].tolist()==train.tolist()
                    pred=saved['model'].predict(d['handcrafted'][test])
                elif method==METHODS[2]:
                    raw=np.load(DATA/'dice_native.npy',mmap_mode='r');owners=np.load(DATA/'dice_native_owners.npy')
                    scaler=np.load(folder/'scaler.npz')
                    x=torch.as_tensor(np.clip((np.asarray(raw,dtype=np.float32)-scaler['mean'])/scaler['sd'],-10,10),device='cuda')
                    model=DICENet().cuda();checkpoint=torch.load(folder/'refit/weights.pt',weights_only=False)
                    assert checkpoint['training_subjects']==train.tolist()
                    model.load_state_dict(checkpoint['state_dict']);p,_=evaluate_windows(model,x,owners,test)
                    assert np.allclose(p,r['details']['probabilities'],atol=2e-5)
                    pred=p.argmax(1);del model,x;torch.cuda.empty_cache()
                else:
                    own=np.load(DATA/'owners.npy');indices=np.flatnonzero(np.isin(own,test));owners=own[indices]
                    templates=np.load(folder/'templates.npz')
                    assert templates['training_subjects'].tolist()==train.tolist()
                    selected=np.load(folder/'selection_templates.npz')
                    assert selected['training_subjects'].tolist()==r['details']['stopping_train']
                    maps=np.load(DATA/'micro_maps.npy',mmap_mode='r')
                    seq=microstate_sequences(maps[indices],templates['templates'])
                    outputs=[]
                    for stage,channels in [(1,[8,9,10,11,0,1,2,3]),(2,[0,1,2,3,4,5,6,7])]:
                        checkpoint=torch.load(folder/f'refit_stage{stage}/weights.pt',weights_only=False)
                        assert checkpoint['training_subjects']==(train if stage==1 else train[y[train]!=2]).tolist()
                        model=MicrostateNFNet().cuda();model.load_state_dict(checkpoint['state_dict'])
                        x=torch.as_tensor(seq[:,channels],device='cuda')
                        outputs.append(evaluate_windows(model,x,owners,test))
                        del model,x;torch.cuda.empty_cache()
                    pd=outputs[0][0][:,1];pf=outputs[1][0][:,1]
                    p=np.column_stack([pd*(1-pf),pd*pf,1-pd])
                    assert np.allclose(p,r['details']['probabilities'],atol=2e-5)
                    pred=np.where(outputs[0][1].argmax(1)==0,2,outputs[1][1].argmax(1))
                assert pred.tolist()==r['predicted'], f'Prediction replay mismatch: {fold}/{method}'
                assert metric(y[test],pred)==r['metrics']
                checked.append({'fold':fold,'method':method,'prediction_replay':True,'test_people':len(test),'overlap':0})
                print(f'PASS fold{fold+1} {method}: {len(test)} saved held-out predictions replayed',flush=True)
    report(out,manifest,y)
    controls={}
    previous=json.loads((ROOT/'neural_comparison/graph_ntf_svm/outputs/full/oof_predictions.json').read_text())
    for method in ['direct','gntf','tt']:
        pred=np.full(174,-1,int);coverage=np.zeros(174,int)
        for r in previous:
            if r['repeat']==0 and r['method']==method:
                assert r['test_indices']==protocol['outer_splits'][r['fold']]['test']
                ix=np.array(r['test_indices']);pred[ix]=np.asarray(r['probabilities']).argmax(1);coverage[ix]+=1
        assert np.all(coverage==1)
        controls[method]=metric(y,pred)
    write_json(out/'matched_previous_controls.json',controls)
    write_json(out/'verification.json',{'complete':True,'participants':174,'folds_per_method':5,
        'methods':METHODS,'replays':checked,'code_hashes_verified':True,'input_hashes_verified':True,
        'each_participant_held_out_once_per_method':True,'cross_participant_duplicate_windows':0,
        'limitation':'Internal validation on a previously explored cohort; adaptations described in README, legacy boundary-handling limitation retained.'})
    print('VERIFIED: all20 saved folds, all174 participants per method, frozen inputs/code and matched controls',flush=True)


if __name__=='__main__':main()
