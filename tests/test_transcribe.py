import os

import whisper


model_name = "tiny"
model = whisper.load_model(model_name).cuda()
audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

language = "en" if model_name.endswith(".en") else None
result = model.transcribe(audio_path, language=language, temperature=0.0, task="translate")
assert result["language"] == "en"

print(result["text"])
transcription = result["text"].lower()
assert "my fellow americans" in transcription
assert "your country" in transcription
assert "do for you" in transcription
