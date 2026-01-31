import torchcrepe
import torch
import resampy
import numpy as np
import librosa

def pitchpred(src, cuda: bool):
    y, sr = librosa.load(src, sr=16000)
    print(sr)

    audio = torch.from_numpy(y).float().unsqueeze(0)

    hop_length = 80

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
    pitch, confidence= torchcrepe.predict(audio,
                            sr,
                            hop_length,
                            fmin,
                            fmax,
                            model,
                            batch_size=batch_size,
                            device=device,
                            return_periodicity=True
                            )

    
    pitch = pitch.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()
    times = np.arange(len(pitch)) * hop_length / sr

    return pitch, confidence, times

def bpmpred(src):
    y, sr = librosa.load(src, sr=16000)
    hop_ms = 0.02
    timestamp_ms = 0.005
    data_multiply = hop_ms / timestamp_ms
    hop_length = sr * hop_ms
    
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

    bpm = np.repeat(target_times, 4)

    return bpm