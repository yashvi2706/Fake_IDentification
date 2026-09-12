# ML Training: Tamper Detection (ELA + ResNet18)

This directory contains the training pipeline for the Core AI Innovation of our SIH project: the Tamper Detection model.

## The Pipeline

We treat forgery detection as a binary classification problem (`authentic` vs. `tampered`).
1. **ELA Transform (`ela.py`)**: Before feeding images to the CNN, they pass through an Error Level Analysis transform. ELA works by resaving the image at a known quality (e.g., 90%) and subtracting the new image from the original. Areas that were spliced or edited will have different compression grids and will light up in the ELA difference image.
2. **CNN Training (`train_tamper_cnn.py`)**: We use a `torchvision` pre-trained `ResNet18` (for fast inference) and modify the final fully connected layer for 2 classes. The model is trained on the ELA-transformed synthetic dataset.

## How to Retrain

1. Ensure you have run the data generator in `../data/`.
2. Install dependencies: `pip install torch torchvision tqdm pillow`
3. Run the training script:
   ```bash
   python train_tamper_cnn.py
   ```
4. The trained weights will be saved to `checkpoints/tamper_resnet18.pth`.

*(Note: The `checkpoints/` directory is gitignored because PyTorch models are large. If running on a restrictive Windows host environment that blocks PyTorch `shm.dll`, it is recommended to run this training script inside a Linux Docker container or WSL2).*
