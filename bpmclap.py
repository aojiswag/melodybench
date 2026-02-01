import numpy as np
import librosa
import soundfile as sf
import pandas as pd


def generate_clap(sr, duration=0.03):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    envelope = np.exp(-t * 80)   # 빠른 감쇠
    clap = noise * envelope
    clap /= np.max(np.abs(clap))
    return clap


def bpmclap(src, dt):
    
    bpm_seq = pd.read_csv("x_bpm.csv", header=None).iloc[:, 3].values
