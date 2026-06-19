import numpy as np
import pandas as pd
import torch

from dmdc import evaluar_horizonte, extraer_latente, persistencia
from modelo_autoencoder import cargar_autoencoder
from preprocesamiento import cargar_config, cargar_dataset, hacer_ventanas, particiones


def main():
    cfg = cargar_config()
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model, ckpt = cargar_autoencoder(cfg)
    df5 = cargar_dataset(cfg)
    i_train, i_val = particiones(len(df5), cfg)
    X_all = (df5.values - np.asarray(ckpt["scaler_mean"])) / np.asarray(ckpt["scaler_scale"])
    X_test = hacer_ventanas(X_all[i_val:], ckpt["window"])
    with torch.no_grad():
        recon, _ = model(torch.FloatTensor(X_test))
    recon = recon.numpy()
    print(f"autoencoder test: MAE={np.mean(np.abs(recon - X_test)):.4f}  "
          f"MSE={np.mean((recon - X_test) ** 2):.4f}")

    d = extraer_latente(cfg)
    m = np.load(cfg["modelo_dmdc"], allow_pickle=True)
    A, B, C, D = m["A"], m["B"], m["C"], m["D"]

    rows = []
    for H in cfg["horizontes"]:
        rv = evaluar_horizonte(d, A, B, C, D, d["k_train"], d["k_val"], H)
        rt = evaluar_horizonte(d, A, B, C, D, d["k_val"], len(d["Z"]), H)
        rows.append([H * 5, rv[0], rv[1], rt[0], rt[1],
                     persistencia(d, d["k_val"], len(d["Z"]), H)])
    tabla = pd.DataFrame(rows, columns=["horizonte_min", "rmse_val", "mae_val",
                                        "rmse_test", "mae_test", "persistencia_test"]).round(2)
    print(tabla.to_string(index=False))
    out = "salidas/metricas_dmdc.csv"
    tabla.to_csv(out, index=False)
    print(f"guardado: {out}")


if __name__ == "__main__":
    main()
