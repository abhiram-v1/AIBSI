import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
import numpy as np
from spectral_data import HERE,AUDIT,POLICY,prepare_one,feature_names,dump,sha


def main():
    dest=HERE/'outputs/prepared';dest.mkdir(parents=True,exist_ok=True)
    ps=[p for p in json.loads((AUDIT/'outputs/prepared/plan.json').read_text())['participants'] if p['source']=='ds004504']
    table=pd.read_csv(HERE.parents[1]/'dataset/ds004504/participants.tsv',sep='\t').set_index('participant_id')
    for p in ps:
        row=table.loc[p['subject_id'].split(':')[-1]]
        assert {'A':'AD','F':'FTD','C':'HC'}[row['Group']]==p['label']
        p['age']=float(row['Age']);p['sex']=str(row['Gender']);p['mmse']=float(row['MMSE'])
    assert len(ps)==88
    dump(dest/'plan.json',{'participants':ps,'policy':POLICY,'participant_table_sha256':sha(HERE.parents[1]/'dataset/ds004504/participants.tsv')})
    with ProcessPoolExecutor(2) as pool:
        futures=[pool.submit(prepare_one,i,p,str(dest)) for i,p in enumerate(ps)]
        for n,f in enumerate(as_completed(futures),1):
            i,status=f.result();print(f'{n}/88 ({n/88:.1%}) {ps[i]["subject_id"]} {status}',flush=True)
    files=[dest/'parts'/f'{i:03d}.npz' for i in range(88)]
    keys=np.load(files[0]).files;arrays={k:np.stack([np.load(f)[k] for f in files]) for k in keys if k!='frequency'}
    arrays['frequency']=np.load(files[0])['frequency'];np.savez_compressed(dest/'features.npz',**arrays)
    dump(dest/'feature_names.json',feature_names(arrays['frequency']))
    print('Fourier feature preparation complete;88 participant samples.',flush=True)


if __name__=='__main__':main()
