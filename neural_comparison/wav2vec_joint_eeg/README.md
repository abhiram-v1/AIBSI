# Joint masked-context EEG classification

This experiment compares three models on the same 174 participants and the same five outer-test subject folds as the first G-NTF repeat. It asks whether masked-context learning and class-contrastive learning help a CNN with an AD/FTD/HC classification head trained from the first optimization step.

The input is the already prepared 19-channel, 24-mel-bin, 13-frame log-mel tensor from `advanced_clean_spectral/outputs/prepared`. It is a time-resolved spectrogram, not the G-NTF four-statistic summary or a raw waveform. The experiment is inspired by wav2vec/BENDR but does not reproduce quantization or raw-waveform pretraining.

## Run

From `C:\Projects\AIBSI` with Python 3.10 and CUDA-enabled PyTorch:

```powershell
py -3.10 -u neural_comparison/wav2vec_joint_eeg/run.py --epochs 35 --seed 20261116 --out neural_comparison/wav2vec_joint_eeg/outputs/full
py -3.10 -u neural_comparison/wav2vec_joint_eeg/run.py --epochs 35 --seed 20261117 --out neural_comparison/wav2vec_joint_eeg/outputs/seed_20261117
py -3.10 -u neural_comparison/wav2vec_joint_eeg/run.py --epochs 35 --seed 20261118 --out neural_comparison/wav2vec_joint_eeg/outputs/seed_20261118
py -3.10 neural_comparison/wav2vec_joint_eeg/summarize.py
```

Each run saves the exact protocol, validation history, one outer-test prediction per participant, fold metrics and checkpoints. Rerunning in the same output directory reuses existing fold results. Use a new directory after changing code or settings.

The diagnosis-only arm minimizes participant-level cross-entropy. The masked arm adds Smooth L1 prediction of a three-frame latent span from surrounding frames. The masked-contrastive arm additionally pulls same-class participant embeddings together and pushes different classes apart. All arms share the same 88,899-parameter CNN and two-layer context Transformer. The classification head trains from epoch one in every arm.

For each outer fold, 20% of its training people form a stratified validation set. The best epoch is selected by participant macro F1 (balanced accuracy and log loss break ties). The remaining outer-test people are evaluated once at that checkpoint. Training samples class-balanced bags of six windows per person; validation and test use all available windows per person. All learned scaling uses only the fold's model-training people. The `RESULTS.md` file under `outputs/` reports the three-seed comparison and its limitations.
