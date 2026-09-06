"""Independent replay of the budget sensitivity and final delivery manifest."""
from pathlib import Path
import json
import joblib
import numpy as np
from prepare import OUT,ROOT,sha
from run import load_inputs,sig,metric
from verify_report import main as verify_main
from threadpoolctl import threadpool_limits


def main():
    verify_main()
    x,controls,subjects,y=load_inputs()
    dest=OUT/'convergence_sensitivity'
    report=json.loads((dest/'report.json').read_text())
    paths=sorted(p for p in dest.glob('*.json') if p.name.startswith(('full_','curve_')))
    assert len(paths)==25
    curve=json.loads((OUT/'learning_curve.json').read_text())['points']
    all_pred={budget:np.full((3,88),-1,dtype=int) for budget in ['100','2000']}
    checked=0
    for path in paths:
        row=json.loads(path.read_text())
        train,test=np.array(row['train']),np.array(row['test'])
        assert row['original_signature']==sig() and not set(train)&set(test)
        base=json.loads((OUT/f'fold_{row["split"]:02d}.json').read_text())
        assert row['test']==base['test'] and set(train)<=set(base['train'])
        assert row['truth']==y[test].tolist()
        if row['fraction']==1: assert row['train']==base['train']
        for budget,result in row['budgets'].items():
            saved=joblib.load(dest/result['model_file'])
            assert saved['signature']==sig()
            assert np.array_equal(saved['train'],train) and np.array_equal(saved['test'],test)
            model=saved['model']
            assert model.penalty==.1 and model.maxiter==int(budget)
            pred=model.predict(x[test]); scores=model.decision_function(x[test])
            assert np.array_equal(pred,result['predictions'])
            np.testing.assert_allclose(scores,result['scores'],atol=1e-10)
            np.testing.assert_allclose(metric(y[test],pred)['balanced_accuracy'],result['metrics']['balanced_accuracy'])
            if row['fraction']==1:
                assert np.all(all_pred[budget][row['split']//5,test]==-1)
                all_pred[budget][row['split']//5,test]=pred
            if budget=='100' and row['split']<5:
                original=next(p for p in curve if p['split']==row['split'] and p['fraction']==row['fraction'])
                assert original['train']==row['train'] and original['predictions']==pred.tolist()
            checked+=1
    assert checked==50
    for budget,p in all_pred.items():
        assert np.all(p>=0)
        np.testing.assert_allclose(np.mean([metric(y,a)['balanced_accuracy'] for a in p]),report['results'][budget]['mean_balanced_accuracy'])
    check=dict(status='PASS',original_model_entries_replayed=90,sensitivity_models_replayed=50,
               sensitivity_subject_splits_checked=25,original_learning_curve_100_iteration_predictions_reproduced=True,
               original_run_signature=sig(),source_arrays_unchanged=True,
               note='Saved-model replay and provenance consistency are distinct from optimization convergence and diagnostic validity.')
    (OUT/'delivery_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json')
    files+=sorted(p for p in Path(__file__).parent.glob('*') if p.is_file())
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(check,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
