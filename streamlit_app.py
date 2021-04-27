from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from pl_bolts.models.self_supervised import Moco_v2

from models import SSLCOVIDNet
from transforms import Moco2EvalCovidxCTTransforms
from utils import auto_body_crop

BACKBONE_CHECKPOINT = "backbone.ckpt"
MODEL_CHECKPOINT = "model.ckpt"
WIDTH, HEIGHT = 330, 330

st.title(
    "An Improvement in COVID-19 Detection on Computed Tomography Images \
    via Self-Supervised Learning")
st.markdown("##")
st.header("Author: Tran Nhat Minh Hoang")


def pred_to_str(pred: int):
    if pred == 0:
        return "Normal"
    elif pred == 1:
        return "Pneumonia"
    else:
        return "COVID-19"


@st.cache
def get_net(backbone_checkpoint, model_checkpoint):
    feature_extractor = Moco_v2.load_from_checkpoint(backbone_checkpoint)
    model = SSLCOVIDNet.load_from_checkpoint(model_checkpoint,
                                             moco_extractor=feature_extractor)
    model.eval()
    return model


try:
    assert Path(BACKBONE_CHECKPOINT).exists(), "FE checkpoint not found."
    assert Path(MODEL_CHECKPOINT).exists(), "Model checkpoint not found."
    model = get_net(BACKBONE_CHECKPOINT, MODEL_CHECKPOINT)
    model.eval()

    sidebar = st.sidebar
    sidebar.title("Sidebar")
    sidebar.selectbox("Choose a demo",
                      ("Exploratory Data Analysis", "COVID-19 Detection"))

    uploaded_image = st.file_uploader("Upload CT-Scan file",
                                      type=["png", "jpg", "jpeg"])

    if not uploaded_image:
        st.info("Please upload a CT-Scan file")
        st.stop()
    else:
        img = Image.open(uploaded_image)
        img_body, _ = auto_body_crop(np.asarray(img))
        img_body = np.stack([img_body] * 3, axis=-1)
        img = Image.fromarray(img_body)
        transforms = Moco2EvalCovidxCTTransforms(height=224)
        img_q, img_k = transforms(img)
        img_q = img_q.unsqueeze(0)

        left_column, right_column = st.beta_columns(2)
        left_column.image(img, caption="CT-Scan sample", width=330)

        output = model(img_q)
        _, pred = torch.max(output, 1)
        result = pred_to_str(pred)
        raw_img = right_column.image(img_body, caption=f"Result: {result}",
                                     width=WIDTH)
except Exception as e:
    st.error(e)
