# Testing bark as an option for the voice integration
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello this is the voice that will be used for the app. How are you? 
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("alex.wav", SAMPLE_RATE, audio_array)
