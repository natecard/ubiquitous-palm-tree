from datetime import datetime
from transformers import AutoModel
import os
import torch as torch
from scipy.io.wavfile import write as write_wav


# Directory to save audio files
test_dir = os.path.join("test_audio")
# Create the directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)
now = datetime.now()
format_time = now.strftime("%m_%d_%h_%m_%s")
audio_file_name = f"{format_time}test.wav"

pipeline = AutoModel.from_pretrained("collabora/whisperspeech")
text = "Hello this is the voice that will be used for the app. How are you?"
audio_tensor = pipeline(prompt=text)

audio_file = write_wav(audio_file_name, 16000, audio_tensor.numpy())
