import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import CVAE  # モデル定義を別ファイルに分ける場合

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
latent_dim = 3
num_classes = 10
model = CVAE(latent_dim, num_classes).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# Streamlit UI
st.title("Conditional Variational Autoencoder (CVAE) Image Generator")
st.write("### 数字画像を生成します")

# サイドバー設定
st.sidebar.header("設定")
label = st.sidebar.selectbox("生成する数字 (0-9)", list(range(10)), index=0)
n_samples = st.sidebar.slider("生成する画像の枚数", 1, 10, 1)

if st.sidebar.button("生成"):
    # 潜在変数のサンプリング
    z = torch.randn(n_samples, latent_dim).to(device)
    y = torch.tensor([label] * n_samples, dtype=torch.long, device=device)
    
    with torch.no_grad():
        x_gen = model.decoder(z, y)
    
    # 画像表示
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        axes[i].imshow(x_gen[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    
    st.pyplot(fig)


    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
