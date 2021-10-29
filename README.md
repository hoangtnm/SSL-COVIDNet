# Final Project <!-- omit in toc -->

## Contents <!-- omit in toc -->

- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Installation

```bash
docker build -t moco -< Dockerfile

# Download checkpoint
# Google Drive:
# https://drive.google.com/drive/folders/1Aof5mbMlBHtxEHtUFUd6_mVtBtujaYXU?usp=sharing
# You need to download `backbon.ckpt` and `model.ckpt` directly to this folder.
```

## Usage

### Pretraining

```shell
docker run --rm -it --gpus '"device=2,3"' \
  --shm-size 16G --privileged \
  -v ~/Downloads:/home/Downloads \
  -v $(pwd):/moco \
  moco python main_pretrain.py \
  --config configs/pretrain.yaml \
  --trainer.gpus 2 \
  --trainer.precision 16 \
  --model.batch_size 256 \
  --model.num_workers 8 \
  --model.softmax_temperature 0.07
```

### Fine-tuning

```bash
python main_pretrain.py \
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
