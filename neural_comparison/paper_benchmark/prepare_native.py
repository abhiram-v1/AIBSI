"""Native-length DICE features and participant-local microstate map caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from scipy.signal import stft, butter, sosfiltfilt, find_peaks
from threadpoolctl import threadpool_limits

from prepare import OUT, REFERENCE, legacy, BANDS, write_json, sha


def dice_features(w):
    ff, _, spectrum = stft(w.astype(float), fs=128, nperseg=256, noverlap=128,
                           boundary=None, padded=False, axis=-1)
    power = np.abs(spectrum)**2
    bp = np.stack([power[:, :, (ff >= lo) & (ff < hi)].sum(2) for lo, hi in BANDS], axis=2)
    bp /= np.maximum(bp.sum(2, keepdims=True), 1e-30)
    z = mne.time_frequency.tfr_array_morlet(w.astype(float), 128, [2,6,10,18,35],
                    n_cycles=[2,4,7,7,7], output='complex', n_jobs=1, verbose='ERROR')
    cohs = []
    for start in range(0, w.shape[-1]-255, 128):
        zz = z[..., start:start+256]
        cross = np.einsum('ncft,ndft->nfcd', zz, zz.conj(), optimize=True)/256
        pp = np.mean(np.abs(zz)**2, axis=-1).transpose(0,2,1)
        coh = np.abs(cross)/np.sqrt(np.maximum(pp[:,:,:,None]*pp[:,:,None,:],1e-30))
        cohs.append(((coh.sum(-1)-np.diagonal(coh,axis1=-2,axis2=-1))/18).transpose(0,2,1))
    return np.stack([bp,np.stack(cohs,axis=-1)],axis=1).astype(np.float32)


def native_dice(args):
    manifest = json.loads(REFERENCE.read_text())
    parts = OUT/'dice_native_parts';parts.mkdir(exist_ok=True)
    for i,p in enumerate(manifest['participants']):
        if i % args.shards != args.shard:
            continue
        dest=parts/f'{i:03d}.npz'
        if dest.exists():
            continue
        r=legacy.Record(p['subject_id'],p['source'],p['site'],p['label'],p['path'])
        data,_=legacy.load_harmonized(r)
        size=31*128
        candidates=[data[:,s:s+size] for s in range(1280,data.shape[1]-1280-size+1,size)
                    if legacy.valid_window(data[:,s:s+size])[0]]
        chosen=legacy.evenly_limit(candidates,10)
        if not chosen:
            raise RuntimeError(f'No native DICE windows: {p["subject_id"]}')
        w=np.stack(chosen)
        features=dice_features(w)
        assert features.shape==(len(w),2,19,5,30)
        np.savez_compressed(dest,dice=features)
        write_json(dest.with_suffix('.json'),{'subject_id':p['subject_id'],'windows':len(w),
            'source_set_sha256':sha(p['path']),'raw_window_sha256':[__import__('hashlib').sha256(a.tobytes()).hexdigest() for a in w]})
        print(f'Native DICE {i+1}/174: {len(w)} windows',flush=True)
    if args.shards>1:
        return
    records=[np.load(parts/f'{i:03d}.npz')['dice'] for i in range(174)]
    np.save(OUT/'dice_native.npy',np.concatenate(records))
    np.save(OUT/'dice_native_owners.npy',np.concatenate([np.full(len(x),i,np.int32) for i,x in enumerate(records)]))
    write_json(OUT/'dice_native_protocol.json',{'participants':174,'window_count':sum(map(len,records)),
        'window_seconds':31,'frames':30,'frame_seconds':2,'hop_seconds':1,'max_windows_per_person':10,
        'frequencies_hz':BANDS,'code_sha256':sha(__file__),
        'adaptations':'128Hz harmonized 1-40Hz source; Morlet correlations averaged per2s frame; 3 output logits; batch_first corrected; native31s windows are distinct from the8s common windows.'})


def microstate_maps():
    w=np.load(OUT/'windows.npy',mmap_mode='r')
    target=OUT/'micro_maps.npy';peak_path=OUT/'micro_peaks.npy'
    if target.exists() and peak_path.exists() and (OUT/'micro_maps_protocol.json').exists():
        return
    maps=np.lib.format.open_memmap(target,mode='w+',dtype=np.float16,shape=(len(w),4,1024,19))
    peaks=np.lib.format.open_memmap(peak_path,mode='w+',dtype=np.float32,shape=(len(w),4,16,19))
    peaks[:]=0
    filters=[None]+[butter(4,band,fs=128,btype='bandpass',output='sos') for band in [(1,4),(4,8),(8,13)]]
    for start in range(0,len(w),64):
        block=np.asarray(w[start:start+64],float)
        for band,sos in enumerate(filters):
            x=block.copy() if sos is None else sosfiltfilt(sos,block,axis=-1)
            x-=x.mean(1,keepdims=True)
            gfp=x.std(1)
            norm=np.sqrt(np.sum(x*x,axis=1))
            normalized=(x/np.maximum(norm[:,None,:],1e-15)).transpose(0,2,1)
            maps[start:start+len(block),band]=normalized
            for j in range(len(block)):
                locations,_=find_peaks(gfp[j],distance=4)
                selected=locations[np.argsort(gfp[j,locations])[-16:]]
                peaks[start+j,band,:len(selected)]=normalized[j,selected]
        if start%640==0:print(f'Microstate map cache {start}/{len(w)}',flush=True)
    maps.flush();peaks.flush()
    write_json(OUT/'micro_maps_protocol.json',{'shape':list(maps.shape),'bands':['broad1-40','delta1-4','theta4-8','alpha8-13'],
        'templates':'NOT fitted here: all fold training happens later','input_sha256':sha(OUT/'windows.npy'),
        'code_sha256':sha(__file__),'representation':'Per-sample centered unit-norm scalp maps;16 GFP peak maps/window; no window joins'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['dice','micro'])
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    mne.set_log_level('ERROR')
    with threadpool_limits(limits=3):
        native_dice(args) if args.command=='dice' else microstate_maps()
