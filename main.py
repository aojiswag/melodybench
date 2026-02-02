
from audio2wav import audio2wav
from ytdownload import ytmp3

import numpy as np
import pynvml
from demucs import demucs
from mir import bpmpred, pitchpred
from bpmclap import bpmclap
import pandas as pd
from pitch2wav import pitch2wav
from scipy.io import wavfile

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

GPUMEM = mem.total // (1024**2)
print(GPUMEM)

pynvml.nvmlShutdown()

CUDA = True if GPUMEM > 1024 * 8 else False

SRC_PATH = "src/"


def menu():
    print(
        "\n\n\n"
        "0. ytmp3 {ytlink} {name}",
        "1. mp32wav {filename}",
        "2. demucs {filename}",
        "3. mir {filename} {stem (bass/drums/vocals/other)}",
        "4. pitch2wav",
        "5. bpmclap {filename}", sep="\n")

def noargs():
    print("\033[95m", "input file")


def main():
    while True:
        menu()
        raw = input(">> ").strip()

        if not raw:
            continue
        
        parts = raw.split()
        cmd = parts[0]
        args = parts[1:]

        if cmd == "exit" or cmd == "quit":
            break

        if cmd == "ytmp3":
            yturl = args[0]
            name = args[1]

            ytmp3(url=yturl, name=name, src_path=SRC_PATH)
        
        if cmd == "mp32wav":
            args = args[0]
            if not args:
                noargs()
                continue
            if args.lower().endswith(".mp3"):
                name = args[:-4]
            else:
                name = args
                args += ".mp3"
            print(args)
            audio2wav(src=f"{SRC_PATH}{args}", dst=f"{SRC_PATH}{name}.wav")

        if cmd == "demucs":
            args = args[0]
            if not args:
                noargs()
                continue

            if args.lower().endswith(".wav"):
                name = args[:-4]
            else:
                name = args
                args += ".wav"
            src_path = f"{SRC_PATH}{args}"
            demucs(src=src_path, cuda=CUDA)

        if cmd == "mir":
            file = args[0]
            stem = args[1]
            dt = 0.005
            
            if not args:
                noargs()
                continue

            if file.lower().endswith(".wav"):
                name = args[0][:-4]
            else:
                name = args[0]
                file = name+".wav"

            stem_path = f"demucsout/htdemucs/{name}/{stem}.wav"
            src_audio = f"{SRC_PATH}{file}"
            pitch, confidence, timestamp = pitchpred(src=stem_path, dt_ms=dt, cuda=CUDA, viterbi_smooth=True)

            bpm = bpmpred(src=src_audio, hop_ms=0.02, dt_ms=dt)
            print(pitch, confidence, timestamp, bpm)
            min_len = min(len(pitch), len(confidence), len(timestamp), len(bpm))

            result = np.stack([
                pitch[:min_len],
                confidence[:min_len],
                timestamp[:min_len],
                bpm[:min_len]
            ], axis=1)
            
            np.savetxt(f"mirout/{name}.csv", result, delimiter=",", fmt="%.6f")

        if cmd == "bpmclap":
            file = args[0]

            if file.lower().endswith(".wav"):
                name = args[0][:-4]
            elif file.lower().endswith(".csv"):
                name = args[0][:-4]
            else:
                name = args[0]
                file = name+".wav"

            src_audio_path = f"{SRC_PATH}/{file}"
            src_mir_path = f"mirout/{name}.csv"

            bpmclap(src_path=src_audio_path, src_mir_path=src_mir_path, dt=0.005)
        
        if cmd == "pitch2wav":
            file = args[0]
            if file.lower().endswith(".csv"):
                name = args[0][:-4]
            else:
                name = args[0]
                file = name+".csv"
            
            df = pd.read_csv(f"mirout/{file}", header=None)

            f0, time = (df.iloc[:, 0].to_numpy(), df.iloc[:, 2].to_numpy())

            y, fs = pitch2wav(f0=f0, time=time)
            
            wavfile.write("f0_harmonics.wav", fs, y)


if __name__ == "__main__":
    main()