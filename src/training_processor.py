import argparse
import os
from pt_datasets.tonality_dataset import TonalityDataset, collate_fn
from models.tonality_classifier import TonalityClassifier
from pt_datasets.ds_processor import data, label_map, transform
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

parser = argparse.ArgumentParser(description="Train and save TonalityClassifier model.")
parser.add_argument('--save_path', type=str, default=os.path.join(os.path.expanduser('~'), 'tonality_classifier.pt'),
                    help='Path to save the trained model (default: ~/tonality_classifier.pt)')
args = parser.parse_args()

# Assume you have a PyTorch Dataset: TonalityDataset
train_loader = DataLoader(TonalityDataset(data, label_map, transform=transform), batch_size=32, shuffle=True, collate_fn=collate_fn)

model = TonalityClassifier(input_dim=128, num_classes=4, in_channels=2)  # Adjust input_dim as needed

trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=1, log_every_n_steps=1)
trainer.fit(model, train_loader)

# Save the model's state_dict
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")
