"""Show what 128 versus 256 Hz changes at a fixed STFT duration."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from features import HERE,COHORT,load_eeg,segment_windows,spectral_power,mel_bank


def main():
    person=json.loads(COHORT.read_text())['participants'][0]
    data,intervals,_=load_eeg(person)
    figures=HERE/'outputs/figures';figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for column,fs in enumerate([128,256]):
        signal=data if fs==256 else resample_poly(data,1,2,axis=-1)
        windows,_=segment_windows(signal,8,intervals,fs)
        segment=windows[:1];cfg={'sampling_hz':fs,'stft_seconds':2,'hop_seconds':.5}
        freq,power=spectral_power(segment,cfg)
        mel=np.log10(np.einsum('mf,wcft->wcmt',mel_bank(freq,24),power,optimize=True)+1e-20)
        mel-=np.median(mel)
        ax=axes[0,column];count=int(.25*fs)
        ax.plot(np.arange(count)/fs,segment[0,8,:count]*1e6,'o-',markersize=3)
        ax.set(xlabel='Time within accepted segment (s)',ylabel='O1 voltage (microvolts)',title=f'{fs} Hz: {count} samples in 0.25 s')
        im=axes[1,column].imshow(mel[0,8],origin='lower',aspect='auto',extent=(.75,7.25,.5,24.5),cmap='magma')
        axes[1,column].set(xlabel='STFT frame center (s)',ylabel='Mel band',title=f'2-second STFT, 0.5-second hop: {mel.shape[-1]} frames')
        fig.colorbar(im,ax=axes[1,column],label='Centered log10 mel power')
    fig.suptitle(f'{person["subject_id"]}: both rates analyse 1–40 Hz; same STFT frequency spacing (0.5 Hz)')
    fig.savefig(figures/'sampling_rate_example.png',dpi=170);plt.close(fig)
    print(figures/'sampling_rate_example.png')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
