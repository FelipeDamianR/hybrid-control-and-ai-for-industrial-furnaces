import json

import numpy as np
import pandas as pd


def cargar_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def encontrar_glitches(s, umbral=5.0):
    return (s.diff().abs() > umbral) & (s.diff(-1).abs() > umbral)


def limpiar(df, umbral=5.0):
    df = df.copy()
    for col in ["sensor_temp_main", "sensor_temp_backup"]:
        g = encontrar_glitches(df[col], umbral)
        df.loc[g, col] = np.nan
        df[col] = df[col].interpolate(method="time", limit_direction="both")
    return df


def cargar_dataset(cfg):
    df = pd.read_csv(cfg["dataset"], parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = limpiar(df, cfg["glitch_threshold"])
    return df[cfg["features"]].resample(cfg["resample"]).mean().interpolate()


def particiones(n, cfg):
    return int(n * cfg["train_frac"]), int(n * cfg["val_frac"])


def hacer_ventanas(X, W):
    m = len(X) - W + 1
    out = np.zeros((m, W, X.shape[1]))
    for i in range(m):
        out[i] = X[i:i + W]
    return out
