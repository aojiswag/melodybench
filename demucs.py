import subprocess

def demucs(src, cuda):

    if cuda == True:
        cmd = [
            "demucs",
            "-d", "cuda",
            "-n", "htdemucs", src,
            "-o", "demucsout",
            "--filename", "{track}/{stem}.wav"
        ]
        print("\033[95m", "demucs run as cuda")
    else:
        cmd = [
            "demucs",
            "-d", "cpu",
            "-n", "htdemucs", src,
            "-o", "demucsout",
            "--filename", "{track}/{stem}.wav"
        ] 
        print("\033[95m", "demucs run as cpu")
    subprocess.run(cmd)
