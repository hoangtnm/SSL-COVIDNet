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
docker run --rm -it --gpus '"device=2,3"' \
  --shm-size 16G --privileged \
  -v ~/Downloads:/home/Downloads \
  -v $(pwd):/moco \
  moco python main_finetune.py \
  --config configs/finetune.yaml \
  --trainer.gpus 1 \
  --trainer.precision 16 \
  --model.pretrained CHECKPOINT \
  --data.random_sampling true
```

### Deployment with Streamlit

```bash
streamlit run streamlit_app.py
```

## References
