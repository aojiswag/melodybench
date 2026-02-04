import torchcrepe
import torch
import resampy
import numpy as np
import librosa
import scipy

import numpy as np
import librosa
import torch
import torchcrepe

import crepe

def clamp_octave_jump(
    f0,
    periodicity,
    max_jump_octaves=0.7,
    conf_thresh=0.3
):
    f0 = f0.copy()

    for t in range(1, len(f0)):
        if not np.isfinite(f0[t]) or not np.isfinite(f0[t-1]):
            continue

        jump = np.log2(f0[t] / f0[t-1])

        if abs(jump) > max_jump_octaves and periodicity[t] < conf_thresh:
            f0[t] = f0[t-1]

    return f0



def viterbi_f0_predict(
    f0,
    periodicity=None,
    alpha=3.0,        # emission 가중치
    beta=2.0,         # transition 가중치
    gamma=0.5,        # harmonic penalty
    fmin=50.0,
    fmax=2000.0
):
    T = len(f0)

    # ---------- 후보 생성 ----------
    candidates = []
    for t in range(T):
        if not np.isfinite(f0[t]) or f0[t] <= 0:
            candidates.append(np.array([np.nan]))
            continue

        cands = [f0[t]]

        # octave-down만 허용
        if f0[t] * 0.5 >= fmin:
            cands.append(f0[t] * 0.5)

        # fmax 근처 후보 제거 (흡수 상태 방지)
        cands = [c for c in cands if c <= 0.9 * fmax]

        candidates.append(np.array(cands))

    # ---------- DP 테이블 ----------
    dp = [np.full(len(c), np.inf) for c in candidates]
    back = [np.zeros(len(c), dtype=int) for c in candidates]
    dp[0][:] = 0.0

    # ---------- harmonic penalty ----------
    def harmonic_penalty(r):
        if abs(np.log2(r)) < 0.03:
            return -gamma          # 유지 보너스
        if abs(np.log2(r / 0.5)) < 0.03:
            return +2 * gamma      # octave-down은 위험
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

                # transition
                trans = abs(np.log2(fj / fi))
                trans_cost = beta * trans

                # emission (periodicity 반영)
                if not np.isfinite(f0[t]):
                    emit = 0.0
                else:
                    conf = 1.0
                    if periodicity is not None:
                        conf = max(periodicity[t], 1e-3)
                    emit = abs(np.log2(fj / f0[t])) / conf

                # 음역 prior (고주파 억제)
                prior_penalty = 0.0
                if fj > 800:
                    prior_penalty = (fj / 800.0) ** 2

                cost = (
                    dp[t - 1][i]
                    + trans_cost
                    + alpha * emit
                    + harmonic_penalty(fj / fi)
                    + prior_penalty
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
    time, f0, periodicity = crepe.predict(audio, sr, viterbi=True)
    print(type(f0))
    f0 = clamp_octave_jump(f0, periodicity)
    f0 = scipy.signal.medfilt(f0, kernel_size=5)

    return time, f0, periodicity

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
