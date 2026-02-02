import torchcrepe
import torch
import resampy
import numpy as np
import librosa


import numpy as np

def viterbi_f0_predict(
    f0,
    alpha=1.0,   # 관측(f0 유지) 가중치
    beta=2.0,    # 연속성 가중치
    gamma=0.5,   # harmonic 보너스
    fmin=50.0,
    fmax=2000.0
):
    T = len(f0)

    # ---------- 후보 생성 ----------
    candidates = []
    for t in range(T):
        if not np.isfinite(f0[t]) or f0[t] <= 0:
            candidates.append([np.nan])
            continue

        cands = np.array([
            f0[t],
            f0[t] * 0.5,
            f0[t] * 2.0,
        ])

        cands = cands[(cands >= fmin) & (cands <= fmax)]
        candidates.append(cands)

    # ---------- DP 테이블 ----------
    dp = [np.full(len(c), np.inf) for c in candidates]
    back = [np.zeros(len(c), dtype=int) for c in candidates]

    dp[0][:] = 0.0

    # ---------- cost 함수 ----------
    def harmonic_penalty(r):
        if abs(np.log2(r)) < 0.03:
            return -gamma       # 유지 보너스
        if abs(np.log2(r / 2)) < 0.03:
            return +1.5*gamma       # octave up
        if abs(np.log2(r / 0.5)) < 0.03:
            return +2.5*gamma    # octave down (더 위험)
        return 0.0

    # ---------- Viterbi ----------
    for t in range(1, T):
        for j, fj in enumerate(candidates[t]):
            if not np.isfinite(fj):
                continue

            best_cost = np.inf
            best_i = 0

            for i, fi in enumerate(candidates[t - 1]):
                if not np.isfinite(fi):
                    continue

                trans = abs(np.log2(fj / fi))

                if abs(trans - 1.0) < 0.04:
                    trans_cost = 0.0     # octave는 거리로는 벌주지 않음
                else:
                    trans_cost = beta * trans

                if not np.isfinite(f0[t]):
                    emit = 0.0
                else:
                    emit = abs(np.log2(fj / f0[t]))

                cost = (
                    dp[t - 1][i]
                    + trans_cost
                    + alpha * emit
                    + harmonic_penalty(fj / fi)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_i = i

            dp[t][j] = best_cost
            back[t][j] = best_i

    # ---------- backtrace ----------
    out = np.zeros(T)
    idx = np.argmin(dp[-1])

    for t in reversed(range(T)):
        out[t] = candidates[t][idx]
        idx = back[t][idx]

    return out


def pitchpred(src, dt_ms, cuda: bool, viterbi_smooth: bool):
    y, sr = librosa.load(src, sr=16000)
    print(sr)

    audio = torch.from_numpy(y).float().unsqueeze(0)

    hop_length = int(sr * dt_ms)

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 2000

    # Select a model capacity--one of "tiny" or "full"
    model = 'full'
    
    if cuda:
        device = 'cuda:0'
        batch_size = 2048
        print("crepe run as cuda")
    else:
        device = 'cpu'
        batch_size = 512
        print("crepe run as cpu")

    # Compute pitch using first gpu
    pitch, periodicity= torchcrepe.predict(audio,
                            sr,
                            hop_length,
                            fmin,
                            fmax,
                            model,
                            batch_size=batch_size,
                            device=device,
                            return_periodicity=True
                            )
    
    periodicity = torchcrepe.threshold.Silence(-65.)(
        periodicity, audio, sr, hop_length
    )

    #periodicity[periodicity < 0.03] = 0
    pitch[periodicity == 0] = float("nan")

    
    pitch = pitch.squeeze(0).cpu().numpy()
    
    if viterbi_smooth:
        f0_smooth = viterbi_f0_predict(pitch)
        pitch[:] = f0_smooth   # 🔥 원본 덮어쓰기

    periodicity = periodicity.squeeze(0).cpu().numpy()
    times = np.arange(len(pitch)) * hop_length / sr

    return pitch, periodicity, times

def bpmpred(src,hop_ms,dt_ms):
    y, sr = librosa.load(src, sr=16000)
    data_multiply = hop_ms / dt_ms
    hop_length = int(sr * hop_ms)
    
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    # ======================
    # 3. Tempogram (BPM vs time)
    # ======================
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
    )
    

    # BPM values for tempogram rows
    bpms = librosa.tempo_frequencies(
        tempogram.shape[0],
        sr=sr,
        hop_length=hop_length
    )
    valid = (bpms > 30) & (bpms < 300)
    bpms = bpms[valid]
    tempogram = tempogram[valid, :]
    # ======================
    # 4. Pick strongest BPM per frame
    # ======================
    bpm_per_frame = bpms[np.argmax(tempogram, axis=0)]

    # Frame timestamps
    frame_times = librosa.frames_to_time(
        np.arange(len(bpm_per_frame)),
        sr=sr,
        hop_length=hop_length
    )

    # ======================
    # 5. Resample to 5ms grid
    # ======================
    target_times = np.arange(0, frame_times[-1], hop_ms)

    bpm = np.repeat(bpm_per_frame, 4)

    return bpm
