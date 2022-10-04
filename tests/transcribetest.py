import os
import torch
import whisper
from whisper import ModelDimensions
from whisper import Whisper
################
model_name = "large"
download_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
checkpoint_file = model_name + ".pt"

with (open(os.path.join(download_root, checkpoint_file), "rb")) as fp:
    checkpoint = torch.load(fp, map_location=None)
del checkpoint_file
"""
encoderDims = dict(checkpoint["dims"].items())
encoderDims['n_vocab'] = 0
encoderDims['n_text_ctx'] = 0
encoderDims['n_text_state'] = 0
encoderDims['n_text_head'] = 0
encoderDims['n_text_layer'] = 0
checkpoint['dims'] = encoderDims

toDeleteKeys = []
for k in checkpoint["model_state_dict"].keys():
    if k.startswith("decoder"):
        toDeleteKeys.append(k)
for k in toDeleteKeys:
    del checkpoint["model_state_dict"][k]
model = WhisperSplit(dims)
model.load_state_dict(checkpoint["model_state_dict"])
"""
model = whisper.load_model(model_name)

##############

audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
#audio_path=r"C:\Users\CHAOSTER\Desktop\tcd\[Vimeo] Steve Cioccolante -  Are Gender Roles a Social Construct or Genetic Design.mp4"
language = "en" if model_name.endswith(".en") else None
result = model.transcribe(audio_path, language=language, temperature=0.0, task="translate", verbose=True)
assert result["language"] == "en"

print(result["text"])
transcription = result["text"].lower()
print(" And so my fellow Americans ask not what your country can do for you ask what you can do for your country.")
