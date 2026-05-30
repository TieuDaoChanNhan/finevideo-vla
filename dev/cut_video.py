import subprocess
import os

FFMPEG_BIN = os.environ.get('FFMPEG_PATH')

def trim_video(input_path, output_path, duration=5):
    command = [
        FFMPEG_BIN,
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    subprocess.run(command, check=True)

# Example usage
trim_video("../videos/7045280-hd_1080_1920_30fps.mp4", "../videos/martial_art.mp4")