import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocesamiento import cargar_config, limpiar


def acf(x, nlags):
    x = x - x.mean()
    var = np.sum(x ** 2)
    return np.array([np.sum(x[: len(x) - k] * x[k:]) / var for k in range(nlags + 1)])


def main():
    cfg = cargar_config()
    fig_dir = cfg["figuras"]
    os.makedirs(fig_dir, exist_ok=True)

    df_raw = pd.read_csv(cfg["dataset"], parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = limpiar(df_raw, cfg["glitch_threshold"])
    df5 = df[cfg["features"]].resample(cfg["resample"]).mean().interpolate()

    n_glitches = {c: int(((df_raw[c].diff().abs() > cfg["glitch_threshold"]) &
                          (df_raw[c].diff(-1).abs() > cfg["glitch_threshold"])).sum())
                  for c in ["sensor_temp_main", "sensor_temp_backup"]}
    print(f"muestras: {len(df_raw)} | tras resample: {len(df5)} | glitches: {n_glitches}")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df5.index, df5["sensor_temp_main"], color="#3a8fb7", lw=0.9, label="sensor_temp_main")
    ax.plot(df5.index, df5["setpoint"], color="#cf545a", lw=0.9, ls="--", label="setpoint")
    ax.set_ylabel("°C")
    ax.set_title("Temperatura del horno vs setpoint (14 dias)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_serie_temperatura.png"), dpi=120)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, col in zip(axes.flat, cfg["features"]):
        ax.hist(df5[col], bins=50, color="#3a8fb7", alpha=0.85)
        ax.set_title(col, fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribuciones (muestreo 5 min)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_distribuciones.png"), dpi=120)

    corr = df5.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    plt.colorbar(im)
    ax.set_title("Matriz de correlacion")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_correlacion.png"), dpi=120)

    nlags = 72
    rho = acf(df5["sensor_temp_main"].values, nlags)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stem(np.arange(nlags + 1) * 5, rho)
    ax.axhline(0.5, color="#cf545a", ls="--", lw=0.9)
    ax.set_xlabel("lag (min)")
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelacion de sensor_temp_main")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_acf.png"), dpi=120)

    print(f"figuras guardadas en {fig_dir}")


if __name__ == "__main__":
    main()
