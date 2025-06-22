import argparse
import os
import torch
from pt_datasets.tonality_dataset import TonalityDataset, collate_fn
from models.tonality_classifier import TonalityClassifier
from pt_datasets.ds_processor import data, label_map, transform
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Test a saved TonalityClassifier model.")
parser.add_argument('--model_path', type=str, default=os.path.join(os.path.expanduser('~'), 'tonality_classifier.pt'),
                    help='Path to the saved model (default: ~/tonality_classifier.pt)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
args = parser.parse_args()

# Prepare test loader (using the same data for demonstration)
test_loader = DataLoader(TonalityDataset(data, label_map, transform=transform), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# Model params must match training
model = TonalityClassifier(input_dim=128, num_classes=4, in_channels=2)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()

# Get a batch and test
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        print("Predictions:", preds.tolist())
        print("True labels:", y.tolist())
        break  # Only test on the first batch 