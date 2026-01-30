import os
from pytubefix import YouTube

def ytmp3(url, name, src_path):
    # 유튜브 영상 URL 입력
    try:
        # YouTube 객체 생성
        yt = YouTube(url)
        # 오디오 스트림만 필터링
        audio_stream = yt.streams.filter(only_audio=True).first()

        # 다운로드 경로 설정
        output_path = src_path

        # 오디오 다운로드
        print("download...")
        audio_file = audio_stream.download(output_path)

        # 확장자 MP3로 변환
        mp3_file = name + ".mp3"
        os.rename(audio_file, src_path+mp3_file)

        print(f"download complete: {mp3_file}")

    except Exception as e:
        print(f"오류 발생: {e}")