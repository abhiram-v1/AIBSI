"""Read original derivative signals and cache a paired 174-person audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from audit_data import HERE,POLICY,dump,people,prepare_one,sha,COHORT,OPERATORS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();dest=HERE/'outputs/prepared';dest.mkdir(parents=True,exist_ok=True)
    ps=people();dump(dest/'plan.json',{'participants':ps,'policy':POLICY,'cohort_sha256':sha(COHORT),'operators_sha256':sha(OPERATORS)})
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(prepare_one,i,p,str(dest)) for i,p in enumerate(ps)]
        for n,f in enumerate(as_completed(futures),1):
            i,status=f.result();print(f'{n}/{len(ps)} ({100*n/len(ps):.1f}%) {ps[i]["subject_id"]} {status}',flush=True)
    paths=[dest/'parts'/f'{i:03d}.npz' for i in range(len(ps))]
    with np.load(paths[0]) as first:keys=list(first.files)
    arrays={}
    for key in keys:arrays[key]=np.stack([np.load(path)[key] for path in paths])
    np.savez_compressed(dest/'features.npz',**arrays)
    print('Prepared all174; each classifier matrix is174x912. No cohort-level normalization has been fitted.',flush=True)


if __name__=='__main__':main()
