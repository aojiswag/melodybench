
from audio2wav import audio2wav
from ytdownload import ytmp3

import pynvml
from demucs import demucs

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

GPUMEM = mem.total // (1024**2)
print(GPUMEM)

pynvml.nvmlShutdown()

src_path = "src/"


def menu():
    print(
        "\n\n\n"
        "0. ytmp3 {ytlink} {name}",
        "1. mp32wav {filename}",
        "2. demucs {filename}",
        "3. mir {filename}",
        "4. pitch2wav", sep="\n")

def noargs():
    print("\033[95m", "파일명을 입력하세요")


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

            cuda = True if GPUMEM > 1024 * 8 else False

            demucs(src=f"{src_path}{args}", cuda=cuda)
        


        

if __name__ == "__main__":
    main()