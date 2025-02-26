import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# モデル定義
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden_activation = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden_activation)
        logvar = self.fc_logvar(hidden_activation)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_latent = torch.cat([latent_vector, label_embedding], dim=1)
        hidden_activation = F.relu(self.fc_hidden(concatenated_latent))
        output_linear = self.fc_out(hidden_activation)
        reconstructed_image = torch.sigmoid(output_linear).view(-1, 1, 28, 28)
        return reconstructed_image

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, image, label):
        mu, logvar = self.encoder(image, label)
        latent_vector = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(latent_vector, label)
        return reconstructed_image, mu, logvar

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
st.sidebar.header("設定")
label = st.sidebar.selectbox("生成する数字 (0-9)", list(range(10)), index=0)
n_samples = st.sidebar.slider("生成する画像の枚数", 1, 10, 1)

if st.button("生成"):
    z = torch.randn(n_samples, latent_dim).to(device)
    y = torch.tensor([label] * n_samples, dtype=torch.long, device=device)
    
    with torch.no_grad():
        x_gen = model.decoder(z, y)
    
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
