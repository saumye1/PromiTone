import torch
from torch.utils.data import Dataset
import torchaudio
import os
from torch.nn.utils.rnn import pad_sequence

class TonalityDataset(Dataset):
    def __init__(self, data, label_map, transform=None):
        """
        Args:
            data: List of dictionaries or DataFrame with keys 'audio' (file name/path) and 'label' (emotion string or int)
            label_map: Dict mapping emotion label names to integer indices
            transform: Optional, function to apply to loaded audio (e.g., MelSpectrogram)
        """
        self.data = data
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get audio file path and label
        item = self.data[idx]
        audio_path = item['audio']['path']
        label = self.label_map[item['label']]
        print(f"audio_path: {audio_path}")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        length = torch.tensor([waveform.size(1)])


        # Apply transform (e.g., MelSpectrogram)
        if self.transform:
            features = self.transform(input_signal=waveform, length=length)
        else:
            features = waveform

        return features, label

def collate_fn(batch):
    features, labels = zip(*batch)
    # If features are tuples (e.g., (features,)), extract the tensor
    features = [f[0] if isinstance(f, tuple) else f for f in features]
    # Always force 2 channels
    target_channels = 2
    features_fixed = []
    for f in features:
        if f.shape[0] == 1:
            f = f.repeat(2, 1, 1)  # mono to stereo
        elif f.shape[0] > 2:
            f = f[:2, ...]  # more than 2 channels, take first 2
        features_fixed.append(f)
    features = features_fixed
    # Find max size for each dimension (rows, time)
    max_rows = max(f.shape[1] for f in features)
    max_len = max(f.shape[2] for f in features)
    padded = []
    for f in features:
        pad_rows = max_rows - f.shape[1]
        pad_time = max_len - f.shape[2]
        # Pad in (time, rows) order: (left, right) for each dim
        # torch.nn.functional.pad pads last dim first, so order is (time, rows)
        f = torch.nn.functional.pad(f, (0, pad_time, 0, pad_rows))
        padded.append(f)
    features = torch.stack(padded)
    labels = torch.tensor(labels)
    return features, labels
