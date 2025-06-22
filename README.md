# PromiTone

## Model: TonalityClassifier

The `TonalityClassifier` is a convolutional neural network for audio-based emotion classification, implemented with PyTorch Lightning.

### Model Architecture

Assuming input shape: `[batch_size, 2, rows, time]` (2 channels, e.g., stereo Mel spectrogram)

| Layer                        | Input Shape                | Output Shape               | Description                       |
|------------------------------|----------------------------|----------------------------|------------------------------------|
| Conv2d (2→32, 3x3, pad=1)    | [B, 2, R, T]               | [B, 32, R, T]              | 2D convolution                    |
| BatchNorm2d(32)              | [B, 32, R, T]              | [B, 32, R, T]              | Batch normalization                |
| ReLU                         | [B, 32, R, T]              | [B, 32, R, T]              | Activation                         |
| Conv2d (32→64, 3x3, pad=1)   | [B, 32, R, T]              | [B, 64, R, T]              | 2D convolution                    |
| BatchNorm2d(64)              | [B, 64, R, T]              | [B, 64, R, T]              | Batch normalization                |
| ReLU                         | [B, 64, R, T]              | [B, 64, R, T]              | Activation                         |
| AdaptiveAvgPool2d((1,1))     | [B, 64, R, T]              | [B, 64, 1, 1]              | Global average pooling             |
| Flatten                      | [B, 64, 1, 1]              | [B, 64]                    | Flatten to vector                  |
| Dropout(0.3)                 | [B, 64]                    | [B, 64]                    | Dropout regularization             |
| Linear(64→num_classes)       | [B, 64]                    | [B, num_classes]           | Fully connected output             |

- `B`: batch size
- `R`: number of rows (e.g., Mel bands)
- `T`: time frames
- `num_classes`: number of emotion classes (e.g., 4)

### Training
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-4)

---

For more details, see `src/models/tonality_classifier.py`.

---

## How to Train and Test the Model

### Training
Train the model and save the weights using:

```bash
python src/training_processor.py --save_path /path/to/save/model.pt
```
- If `--save_path` is not provided, the model will be saved to your home directory as `~/tonality_classifier.pt`.

### Testing
Test a saved model on a batch of data:

```bash
python src/test_model.py --model_path /path/to/save/model.pt --batch_size 8
```
- If `--model_path` is not provided, it will look for `~/tonality_classifier.pt`.
- The script prints predicted and true labels for a batch.

---

For more advanced evaluation or custom data, modify the scripts as needed. 