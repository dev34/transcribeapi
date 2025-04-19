from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import ffmpeg

app = Flask(__name__)

if not os.path.exists("/usr/bin/ffmpeg"):
    os.system("apt update && apt install -y ffmpeg")

model = whisper.load_model("base")



def extract_audio(video_path: str, audio_path: str):
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(quiet=True, overwrite_output=True)
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {e}")

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if not file.filename.endswith(('.mp4', '.mkv', '.mov', '.avi')):
        return jsonify({'error': 'Unsupported file type'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, file.filename)
        audio_path = os.path.join(tmpdir, "audio.wav")

        # Save uploaded file
        file.save(video_path)

        try:
            extract_audio(video_path, audio_path)
            result = model.transcribe(audio_path)
            return jsonify({'transcription': result['text']})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
