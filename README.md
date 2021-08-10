# Final Project <!-- omit in toc -->

## Contents <!-- omit in toc -->

- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Installation

```bash
# Download the latest version of Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run the installer and follow any required confirmations
bash ./Miniconda3-latest-Linux-x86_64.sh

# Create a virtual Python environment
# to avoid system conflict.
conda create -n final-project \
    python=3.8 \
    pytorch=1.8 \
    pytorch-lightning=1.2.6 \
    torchvision \
    cudatoolkit=11.1 \
    pandas \
    scikit-learn \
    streamlit \
    matplotlib \
    jupyterlab=3 \ 
    plotly

# Activate the environment when using the system
conda activate final-project

# Install some library dependencies
pip install --upgrade protobuf
pip install lightning-bolts opencv-python
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist 
    --upgrade nvidia-dali-cuda110

# Download checkpoint
# Google Drive:
# https://drive.google.com/drive/folders/1Aof5mbMlBHtxEHtUFUd6_mVtBtujaYXU?usp=sharing

# You need to download `backbon.ckpt` and `model.ckpt` directly to this folder.
```

## Usage

### Pretraining

```bash
python main.py \
  --phase pretrain \
  --base_encoder densenet121 \
  --emb_dim 256 \
  --data_dir ../archive \
  --gpus 1 \
  --batch_size 64 \
  --num_workers 4 \
  --random_sampling
```

### Fine-tuning

```bash
python main.py \
  --phase finetune \
  --base_encoder densenet121 \
  --emb_dim 256 \
  --data_dir ../archive \
  --gpus 1 \
  --batch_size 128 \
  --num_workers 4 \
  --random_sampling \
  --base_encoder_checkpoint ./lightning_logs/version_1/checkpoints/densenet121epoch=21-val_loss=1.42.ckpt \
  --max_epochs 10
```

### Deployment with Streamlit

```bash
streamlit run streamlit_app.py
```

## References
