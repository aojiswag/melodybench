import numpy as np
from scipy.interpolate import interp1d
from scipy.io import wavfile

def pitch2wav(time, f0, fs=48000, max_harm=6, rolloff=1.1, gain=0.9):
    # 전체 길이
    duration = time[-1]
    N = int(duration * fs)
    t = np.linspace(0, duration, N)

    # NaN -> 0 (무성구간 처리)
    f0 = np.nan_to_num(f0, nan=0.0)

    # f0를 샘플레이트에 맞춰 보간
    interp = interp1d(time, f0, kind="linear", fill_value=0.0, bounds_error=False)
    f0_sampled = interp(t)

    # 위상 누적
    phase = 2*np.pi*np.cumsum(f0_sampled)/fs

    # 하모닉 합성
    y = np.zeros_like(phase)
    for k in range(1, max_harm+1):
        mask = (k*f0_sampled) < (0.45*fs)  # alias 방지
        ak = 1.0/(k**rolloff)
        y += np.sin(k*phase) * ak * mask

    # 정규화
    peak = np.max(np.abs(y)) + 1e-9
    y = (y/peak) * gain

    return y.astype(np.float32), fs


