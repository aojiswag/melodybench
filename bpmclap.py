import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import os


def generate_clap(sr, duration=0.03):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    envelope = np.exp(-t * 80)   # 빠른 감쇠
    clap = noise * envelope
    clap /= np.max(np.abs(clap))
    return clap

def bpm_to_beat_times(bpm_seq, dt):
    beat_times = []
    t = 0.0
    acc = -0.94

    for bpm in bpm_seq:
        if bpm <= 0:
            t += dt
            continue

        interval = 60.0 / bpm
        acc += dt

        if acc >= interval:
            beat_times.append(t)
            acc -= interval

        t += dt

    return np.array(beat_times)

def add_claps(audio, sr, beat_times, clap):
    out = audio.copy()

    for bt in beat_times:
        idx = int(bt * sr)
        if idx + len(clap) < len(out):
            out[idx:idx+len(clap)] += clap

    # clipping 방지
    out /= max(1.0, np.max(np.abs(out)))
    return out

def bpmclap(src_path, src_mir_path, dt):
    
    name = os.path.splitext(os.path.basename(src_path))[0]

    # 1. load audio
    y, sr = librosa.load(src_path, sr=None)
    bpm_seq = pd.read_csv(src_mir_path, header=None).iloc[:, 3].values
    
    # 2. clap
    clap = generate_clap(sr)

    # 3. BPM → beat times
    beat_times = bpm_to_beat_times(bpm_seq, dt)

    # 4. mix
    y_out = add_claps(y, sr, beat_times, clap)

    # 5. save
    sf.write(f"bpmclap/{name}.wav", y_out, sr)
    
