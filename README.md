# T2VID-Net: Unobtrusive Face Recognition Enhancement for Round-the-Clock Factory Safety Monitoring via Spatial-Aware Cross-Thermal-Visible Synthesis

T2VID-Net is a diffusion-based framework designed to bridge the modality gap between thermal and visible imagery. By jointly optimizing a differentiable spatial alignment mechanism, high-order angular margin constraints, and a dual-level perceptual and structural compensation mechanism, it provides reliable, unobtrusive face recognition for unconstrained industrial environments.

## 📂 Repository Structure

```text
.
├── preprocess_test.py         # Data preprocessing script for testing and evaluation
├── run_diff.sh                # Shell script for executing testing/inference workflows
├── core/                      # Core utilities and evaluation metrics
│   ├── environment.txt        # PIP environment dependencies
│   ├── environment.yml        # Conda environment configuration
│   ├── face_features.py       # Feature extraction module (e.g., ArcFace integration)
│   ├── logger.py              # Standard logging utility
│   ├── metrics.py             # Evaluation metrics (SSIM, PSNR, LPIPS, etc.)
│   └── wandb_logger.py        # Weights & Biases integration for experiment tracking
├── guided_diffusion/          # Core Diffusion Model Architecture (based on DDPM)
│   ├── dist_util.py           # Distributed training utilities
│   ├── fp16_util.py           # Mixed-precision training utilities
│   ├── gaussian_diffusion.py  # Gaussian diffusion process definitions
│   ├── image_datasets.py      # Dataloader and dataset definitions
│   ├── logger.py              # Diffusion-specific logging
│   ├── losses.py              # Loss functions (MSE, Perceptual, Structural, Identity)
│   ├── nn.py                  # Neural network base operations
│   ├── resample.py            # Diffusion timestep resampling schedules
│   ├── respace.py             # Spaced diffusion for faster sampling
│   ├── script_util.py         # Argument parsing and model creation helpers
│   ├── test_diff.py           # Core inference and sampling logic
│   ├── train_util.py          # Training loop and optimization steps
│   ├── unet.py                # U-Net backbone architecture
│   └── valdata.py             # Validation dataset handling
└── scripts/                   # Execution entry points
    ├── T2V_test.py            # Entry script for evaluating the trained model
    └── T2V_train.py           # Entry script for training T2VID-Net from scratch

```

## ⚙️ Environment Setup

We recommend using Anaconda to manage the environment. You can quickly set up the required dependencies using the provided environment files in the `core/` directory.

```bash
# Clone the repository
git clone https://github.com/YourUsername/T2VID-Net.git
cd T2VID-Net

# Create and activate the Conda environment
conda env create -f core/environment.yml
conda activate t2vid

# Alternatively, using pip:
pip install -r core/environment.txt

```

## 📊 Data Preparation

The model is trained and evaluated on the **TFW** and **Tufts** datasets.

1. Download the datasets from their official sources.
2. Place the paired thermal and visible images in your designated `data/` directory.
3. Run the preprocessing script to align and normalize the inputs (ensure paths are configured inside the script):

```bash
python preprocess_test.py --data_dir /path/to/raw/data --save_dir /path/to/processed/data

```

## 🙏 Acknowledgments

This repository borrows partial code from the standard DDPM implementation. We thank the authors for their foundational work. We also gratefully acknowledge the creators of the TFW dataset and the Tufts Face Database.
