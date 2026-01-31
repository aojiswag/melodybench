
from audio2wav import audio2wav
from ytdownload import ytmp3

import pynvml
from demucs import demucs
from mir import bpmpred, pitchpred

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

GPUMEM = mem.total // (1024**2)
print(GPUMEM)

pynvml.nvmlShutdown()

CUDA = True if GPUMEM > 1024 * 8 else False

src_path = "src/"


def menu():
    print(
        "\n\n\n"
        "0. ytmp3 {ytlink} {name}",
        "1. mp32wav {filename}",
        "2. demucs {filename}",
        "3. mir {filename} {stem (bass/drums/vocals/other)}",
        "4. pitch2wav", sep="\n")

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

            ytmp3(url=yturl, name=name, src_path=src_path)
        
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
            audio2wav(src=f"{src_path}{args}", dst=f"{src_path}{name}.wav")

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
            src_path = f"{src_path}{args}"
            demucs(src=src_path, cuda=CUDA)

        if cmd == "mir":
            name = args[0]
            stem = args[1]
            
            if not args:
                noargs()
                continue

            if args.lower().endswith(".wav"):
                name = args[:-4]
            else:
                name = args
                args += ".wav"

            stem_path = f"htdemucs/{name}/{stem}.wav"
            src_audio = f"{src_path}{args}"

            pitch, confidence, timestamp = pitchpred(stem_path, cuda=CUDA)

            bpm = bpmpred(src_audio)

            
            
            


if __name__ == "__main__":
    main()