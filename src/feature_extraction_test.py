import nemo.collections.asr as nemo_asr
import torch
import torchaudio

# Load audio file
waveform, sample_rate = torchaudio.load('audio.wav')

# Use NeMo's MelSpectrogram feature extractor
mel_spec_extractor = nemo_asr.modules.AudioToMelSpectrogramPreprocessor()
mel_spec = mel_spec_extractor(waveform)
