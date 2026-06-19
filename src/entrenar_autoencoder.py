import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from modelo_autoencoder import LSTMAutoencoder
from preprocesamiento import cargar_config, cargar_dataset, hacer_ventanas, particiones


def main():
    cfg = cargar_config()
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    df5 = cargar_dataset(cfg)
    n = len(df5)
    i_train, i_val = particiones(n, cfg)
    W = cfg["window"]

    scaler = StandardScaler().fit(df5.iloc[:i_train].values)
    X_all = scaler.transform(df5.values)
    X_train = hacer_ventanas(X_all[:i_train], W)
    X_val = hacer_ventanas(X_all[i_train:i_val], W)
    X_test = hacer_ventanas(X_all[i_val:], W)

    model = LSTMAutoencoder(len(cfg["features"]), hidden=cfg["hidden"],
                            latent=cfg["latent"], window=W)
    X_tr = torch.FloatTensor(X_train)
    X_va = torch.FloatTensor(X_val)
    X_te = torch.FloatTensor(X_test)
    loader = DataLoader(TensorDataset(X_tr), batch_size=cfg["batch_size"], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.L1Loss()

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for (xb,) in loader:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            recon_v, _ = model(X_va)
            val_loss = loss_fn(recon_v, X_va).item()
        print(f"ep {ep:2d}: train MAE={np.mean(losses):.4f}  val MAE={val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        recon_te, _ = model(X_te)
        test_mae = loss_fn(recon_te, X_te).item()
    print(f"test MAE={test_mae:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": cfg["features"],
        "window": W,
        "latent_dim": cfg["latent"],
    }, cfg["checkpoint"])
    print(f"guardado: {cfg['checkpoint']}")


if __name__ == "__main__":
    main()
