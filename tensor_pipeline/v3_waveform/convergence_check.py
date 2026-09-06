"""Post-hoc optimization-budget sensitivity, NOT a new model-selection sweep.

Keep the predeclared learning-curve configuration (penalty .1). Compare 100
versus 2000 iterations on exactly the same subject splits. Preserve originals.
"""
import json
import time
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from prepare import OUT,sha
from run import load_inputs,sig,metric
from wave_models import SupervisedTT,weights


def main():
    x,controls,subjects,y = load_inputs()
    dest = OUT/'convergence_sensitivity'; dest.mkdir(exist_ok=True)
    protocol = dict(original_signature=sig(),check_code_sha=sha(Path(__file__)),
                    reason='All 120 original supervised fits hit their fixed iteration limit. A training-only pilot also hit 2000; quantify budget sensitivity without selecting the better test score.',
                    status='Post-hoc sensitivity after original outcomes; does not replace original primary endpoint.',
                    fixed_penalty=.1,budgets=[100,2000],new_hyperparameter_selection=False,
                    design='Same 15 full-training outer splits, plus the 10 smaller first-repeat learning-curve subsets.',
                    test_scores_not_used_to_change_protocol=True)
    p = dest/'protocol.json'
    if p.exists(): assert json.loads(p.read_text())==protocol
    else: p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    jobs = []
    for split in range(15):
        f = json.loads((OUT/f'fold_{split:02d}.json').read_text())
        jobs.append(dict(name=f'full_{split:02d}',split=split,fraction=1.,train=f['train'],test=f['test']))
    curve = json.loads((OUT/'learning_curve.json').read_text())
    for r in curve['points']:
        if r['fraction']<1: jobs.append(dict(name=f'curve_{r["split"]:02d}_{int(r["fraction"]*100)}',split=r['split'],fraction=r['fraction'],train=r['train'],test=r['test']))
    records=[]
    for j,job in enumerate(jobs):
        path = dest/(job['name']+'.json')
        if path.exists(): records.append(json.loads(path.read_text())); continue
        start=time.perf_counter()
        train,test = np.array(job['train']),np.array(job['test'])
        assert not set(train)&set(test)
        w=weights(y[train]); np.testing.assert_allclose([w[y[train]==c].sum() for c in range(3)],len(train)/3)
        row=dict(**job,original_signature=sig(),truth=y[test].tolist(),budgets={})
        for budget in [100,2000]:
            m=SupervisedTT(penalty=.1,maxiter=budget).fit(x[train],y[train])
            pred=m.predict(x[test]); scores=m.decision_function(x[test])
            file=dest/f'{job["name"]}_{budget}.joblib'
            joblib.dump(dict(model=m,train=train,test=test,signature=sig()),file)
            loaded=joblib.load(file)
            assert np.array_equal(loaded['model'].predict(x[test]),pred)
            np.testing.assert_allclose(loaded['model'].decision_function(x[test]),scores,atol=1e-10)
            row['budgets'][str(budget)]=dict(predictions=pred.tolist(),scores=scores.tolist(),metrics=metric(y[test],pred),
                                            training_metrics=metric(y[train],m.predict(x[train])),optimization=m.optimization_,model_file=file.name)
        row['seconds']=time.perf_counter()-start
        path.write_text(json.dumps(row,indent=2),encoding='utf-8')
        records.append(row)
        print(f'Budget check {j+1}/{len(jobs)} saved: {job["name"]} in {row["seconds"]:.1f}s.',flush=True)
    summary={}
    for budget in ['100','2000']:
        reps=[]
        for repeat in range(3):
            fs=[r for r in records if r['fraction']==1 and r['split']//5==repeat]
            truth=np.concatenate([r['truth'] for r in fs]); pred=np.concatenate([r['budgets'][budget]['predictions'] for r in fs])
            reps.append(metric(truth,pred))
        lc=[]
        for frac in [.4,.7,1.]:
            fs=[r for r in records if r['fraction']==frac and r['split']<5]
            truth=np.concatenate([r['truth'] for r in fs]); pred=np.concatenate([r['budgets'][budget]['predictions'] for r in fs])
            lc.append(dict(fraction=frac,metrics=metric(truth,pred)))
        opt=[r['budgets'][budget]['optimization'] for r in records]
        summary[budget]=dict(repeats=reps,mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reps])),
                            sd_balanced_accuracy=float(np.std([r['balanced_accuracy'] for r in reps],ddof=1)),
                            mean_accuracy=float(np.mean([r['accuracy'] for r in reps])),
                            learning_curve=lc,converged=sum(o['success'] for o in opt),fits=len(opt),
                            statuses=dict(Counter(o['message'] for o in opt)))
    report=dict(protocol=protocol,results=summary,total_seconds=sum(r['seconds'] for r in records),
                checks=dict(preserved_original_models=True,all_50_models_replayed=True,subject_disjoint=True),
                interpretation='Budget sensitivity only; neither result is a new nested winner or an externally validated performance estimate.')
    (dest/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Supervised TT: optimization-budget sensitivity','',
           'This is a post-hoc diagnostic motivated by all original supervised fits exhausting 100 iterations. '
           'It preserves the original experiment and does not replace its primary score. The head penalty is fixed at '
           '0.1, already specified for the original learning curve; no model setting is selected using these outcomes.','',
           '| Iteration cap | Mean balanced accuracy ± repeat SD | Ordinary accuracy | Optimizer convergence across 25 fits |',
           '|---|---:|---:|---:|']
    for budget,r in summary.items():
        lines.append(f'| {budget} | {100*r["mean_balanced_accuracy"]:.1f}% ± {100*r["sd_balanced_accuracy"]:.1f} pp | {100*r["mean_accuracy"]:.1f}% | {r["converged"]}/{r["fits"]} |')
    lines += ['', 'The accuracy rows use all 88 held-out subjects once per repetition, across the same three five-fold repetitions. '
              'The 25 optimization counts also include ten smaller training subsets. SD is not a confidence interval.','',
              '| Training fraction | 100-iteration held-out BA | 2000-iteration held-out BA |','|---|---:|---:|']
    for a,b in zip(summary['100']['learning_curve'],summary['2000']['learning_curve']):
        lines.append(f'| {100*a["fraction"]:.0f}% | {100*a["metrics"]["balanced_accuracy"]:.1f}% | {100*b["metrics"]["balanced_accuracy"]:.1f}% |')
    lines += ['', 'Learning-curve rows use only the first CV repetition. They cannot establish a required sample size. '
              'The fixed-model 100-iteration mean differs from the original nested supervised-family mean because '
              'that original family could select penalty 0.01 or 0.1 inside each training fold.','',
              'All 50 fitted models and their scores were saved and replayed. Subject memberships and class-weight totals '
              'were checked. A convergence flag, when present, does not establish a global optimum for this nonconvex model. '
              'Remaining budget exhaustion is explicitly reported rather than hidden.','',
              'No further settings were changed after examining this sensitivity result. '
              'See [the original full experiment](../EXPERIMENT_REPORT.md) for the primary endpoint and all other methods.','']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in r.items() if n not in ['repeats','learning_curve']} for k,r in summary.items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
