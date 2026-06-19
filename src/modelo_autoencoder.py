import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=32, latent=8, window=12):
        super().__init__()
        self.window = window
        self.latent = latent
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        h0 = self.from_latent(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_input = h0.transpose(0, 1).repeat(1, self.window, 1)
        out, _ = self.decoder(dec_input, (h0, c0))
        return self.head(out), z


def cargar_autoencoder(cfg):
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = LSTMAutoencoder(len(ckpt["features"]), hidden=cfg["hidden"],
                            latent=ckpt["latent_dim"], window=ckpt["window"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
