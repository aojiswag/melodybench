from pydub import AudioSegment

def audio2wav(src, dst):
    sound = AudioSegment.from_file(src)
    sound.export(dst, format="wav")
    