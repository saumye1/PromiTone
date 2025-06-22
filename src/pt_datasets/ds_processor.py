from .tonality_dataset import TonalityDataset
from datasets import load_dataset

# Example for TrainingDataPro/speech-emotion-recognition-dataset
dataset = load_dataset("TrainingDataPro/speech-emotion-recognition-dataset", split="train")#, download_mode="force_redownload")
emotion_labels = ['euphoric', 'joyfully', 'sad', 'surprised']
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

# Prepare list of dicts with 'audio' and 'label' (20 x 4 = 80 samples)
data = []
for row in dataset:
    for emotion in emotion_labels:
        data.append({'audio': row[emotion], 'label': emotion})


# Optional: define a transform, e.g., MelSpectrogram
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
transform = AudioToMelSpectrogramPreprocessor()

# Create dataset
tonality_dataset = TonalityDataset(data, label_map, transform=transform)
