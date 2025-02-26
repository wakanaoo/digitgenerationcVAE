import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

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
        reconstructed_image_flat = torch.sigmoid(output_linear)
        reconstructed_image = reconstructed_image_flat.view(-1, 1, 28, 28)
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
model = CVAE(latent_dim=3, num_classes=10).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# Streamlit アプリ
st.title("Conditional Variational Autoencoder (CVAE)")

st.write("数字ラベル（0-9）を選択して画像を生成")

digit = st.number_input("生成したい数字を選んでください:", min_value=0, max_value=9, step=1, value=0)

if st.button("画像を生成"):
    z_random_vector = torch.randn(1, 3).to(device)
    label = torch.tensor([digit], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_image = model.decoder(z_random_vector, label)

    img = generated_image.squeeze().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
