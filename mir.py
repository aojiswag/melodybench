import torchcrepe
import torch
import resampy
import numpy as np
import librosa


import numpy as np
def viterbi_f0_predict(
    f0,
    periodicity=None,
    alpha=3.0,
    beta=2.0,
    gamma=0.5,
    fmin=50.0,
    fmax=2000.0,
    octave_penalty=0.5
):
    T = len(f0)

    # ---------- 후보 생성 ----------
    candidates = []
    for t in range(T):
        cands = set()

        if np.isfinite(f0[t]) and f0[t] > 0:
            # 관측 기반
            cands.add(f0[t])
            if f0[t] * 0.5 >= fmin:
                cands.add(f0[t] * 0.5)

            # octave-up은 매우 보수적으로
            if (
                t > 0 and np.isfinite(f0[t - 1])
                and abs(np.log2(f0[t] / f0[t - 1])) < 0.4
                and f0[t] * 2.0 <= fmax
            ):
                cands.add(f0[t] * 2.0)

        # ⭐ 이전 프레임 기반 (복귀 경로 핵심)
        if t > 0 and np.isfinite(f0[t - 1]) and f0[t - 1] > 0:
            prev = f0[t - 1]
            if fmin <= prev <= fmax:
                cands.add(prev)
            if prev * 0.5 >= fmin:
                cands.add(prev * 0.5)

        if len(cands) == 0:
            candidates.append(np.array([np.nan]))
        else:
            candidates.append(np.array(sorted(cands)))

    # ---------- DP 테이블 ----------
    dp = [np.full(len(c), np.inf) for c in candidates]
    back = [np.zeros(len(c), dtype=int) for c in candidates]
    dp[0][:] = 0.0

    # ---------- harmonic penalty ----------
    def harmonic_penalty(r):
        lr = abs(np.log2(r))
        if lr < 0.03:
            return -gamma
        if abs(lr - 1.0) < 0.03:
            return +gamma
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
                if abs(trans - 1.0) < 0.04:
                    trans_cost = beta * trans * octave_penalty
                else:
                    trans_cost = beta * trans

                # emission (periodicity 반영)
                if not np.isfinite(f0[t]):
                    emit = 0.0
                else:
                    conf = 1.0
                    if periodicity is not None and t < len(periodicity):
                        conf = max(periodicity[t], 1e-3)
                    emit = abs(np.log2(fj / f0[t])) / conf

                # fmax 흡수 방지
                edge_penalty = 0.0
                if fj > 0.85 * fmax:
                    edge_penalty = 3.0 * (fj / fmax) ** 2

                cost = (
                    dp[t - 1][i]
                    + trans_cost
                    + alpha * emit
                    + harmonic_penalty(fj / fi)
                    + edge_penalty
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
    periodicity = periodicity.squeeze(0).cpu().numpy()

    if viterbi_smooth:
        f0_smooth = viterbi_f0_predict(f0=pitch, periodicity=periodicity)
        pitch[:] = f0_smooth   # 🔥 원본 덮어쓰기

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
