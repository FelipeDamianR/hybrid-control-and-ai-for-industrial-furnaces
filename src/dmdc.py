import numpy as np
import torch

from modelo_autoencoder import cargar_autoencoder
from preprocesamiento import cargar_config, cargar_dataset, hacer_ventanas, particiones


def extraer_latente(cfg):
    model, ckpt = cargar_autoencoder(cfg)
    df5 = cargar_dataset(cfg)
    W = ckpt["window"]
    X_all = (df5.values - np.asarray(ckpt["scaler_mean"])) / np.asarray(ckpt["scaler_scale"])
    Xw = hacer_ventanas(X_all, W)
    with torch.no_grad():
        Z = np.vstack([model(torch.FloatTensor(Xw[i:i + 512]))[1].numpy()
                       for i in range(0, len(Xw), 512)])
    i_train, i_val = particiones(len(df5), cfg)
    k_train, k_val = i_train - (W - 1), i_val - (W - 1)
    y = df5[cfg["target"]].values[W - 1:]
    U = X_all[W - 1:, [cfg["features"].index(c) for c in cfg["inputs"]]]
    return {
        "Z": Z, "U": U, "y": y, "t": df5.index[W - 1:],
        "k_train": k_train, "k_val": k_val,
        "z_mean": Z[:k_train].mean(0), "u_mean": U[:k_train].mean(0),
        "y_mean": y[:k_train].mean(),
        "scaler_scale": np.asarray(ckpt["scaler_scale"]),
        "latent": ckpt["latent_dim"],
    }


def ajustar(d, lam, latent):
    Zd = d["Z"] - d["z_mean"]
    Ud = d["U"] - d["u_mean"]
    yd = d["y"] - d["y_mean"]
    k = d["k_train"]
    X = np.hstack([Zd[:k - 1], Ud[:k - 1]])
    Theta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Zd[1:k])
    A, B = Theta[:latent].T, Theta[latent:].T
    Xo = np.hstack([Zd[:k], Ud[:k]])
    Phi = np.linalg.solve(Xo.T @ Xo + lam * np.eye(Xo.shape[1]), Xo.T @ yd[:k])
    return A, B, Phi[:latent].reshape(1, -1), Phi[latent:].reshape(1, -1)


def rollout(A, B, C, D, z0, U_seq, latent):
    zs = np.zeros((len(U_seq), latent))
    zs[0] = z0
    for k in range(len(U_seq) - 1):
        zs[k + 1] = A @ zs[k] + B @ U_seq[k]
    return zs @ C.ravel() + U_seq @ D.ravel()


def evaluar_horizonte(d, A, B, C, D, k0, k1, H):
    Zd = d["Z"] - d["z_mean"]
    Ud = d["U"] - d["u_mean"]
    errs = []
    for k in range(k0, k1 - H):
        yp = rollout(A, B, C, D, Zd[k], Ud[k:k + H], d["latent"]) + d["y_mean"]
        errs.append(yp - d["y"][k:k + H])
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def persistencia(d, k0, k1, H):
    y = d["y"]
    e = np.concatenate([y[k] - y[k:k + H] for k in range(k0, k1 - H)])
    return float(np.sqrt(np.mean(e ** 2)))


def seleccionar_lambda(d, cfg):
    candidatos = []
    for lam in cfg["ridge_lambdas"]:
        A, B, C, D = ajustar(d, lam, d["latent"])
        maxev = np.abs(np.linalg.eigvals(A)).max()
        r24, _ = evaluar_horizonte(d, A, B, C, D, d["k_train"], d["k_val"], 288)
        print(f"lambda={lam:6.0f}  max|eig|={maxev:.4f}  RMSE val 24h={r24:7.2f}")
        candidatos.append((lam, maxev, r24))
    return min((c for c in candidatos if c[1] < 1), key=lambda c: c[2])[0]


def main():
    cfg = cargar_config()
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    d = extraer_latente(cfg)
    lam = seleccionar_lambda(d, cfg)
    print(f"lambda seleccionado: {lam}")
    A, B, C, D = ajustar(d, lam, d["latent"])
    u_scale = np.array([d["scaler_scale"][cfg["features"].index(c)] for c in cfg["inputs"]])
    np.savez(cfg["modelo_dmdc"], A=A, B=B, C=C, D=D, dt=float(cfg["dt_segundos"]),
             ridge_lambda=lam, inputs=np.array(cfg["inputs"]), z_mean=d["z_mean"],
             u_mean=d["u_mean"], y_mean=d["y_mean"], u_scale=u_scale)
    eig = np.linalg.eigvals(A)
    print(f"polos |z|: {np.sort(np.abs(eig))[::-1].round(4)}")
    print(f"guardado: {cfg['modelo_dmdc']}")


if __name__ == "__main__":
    main()
