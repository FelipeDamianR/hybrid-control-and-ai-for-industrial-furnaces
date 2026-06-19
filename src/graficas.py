import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

import modelo_control
from dmdc import extraer_latente, rollout
from preprocesamiento import cargar_config


def main():
    cfg = cargar_config()
    fig_dir = cfg["figuras"]
    os.makedirs(fig_dir, exist_ok=True)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    d = extraer_latente(cfg)
    m = np.load(cfg["modelo_dmdc"], allow_pickle=True)
    A, B, C, D, dt = m["A"], m["B"], m["C"], m["D"], float(m["dt"])
    Zd = d["Z"] - d["z_mean"]
    Ud = d["U"] - d["u_mean"]
    y, t, latent = d["y"], d["t"], d["latent"]
    k_train, k_val = d["k_train"], d["k_val"]

    H = 36
    y_pred = np.full(len(Zd), np.nan)
    for k0 in range(0, len(Zd), H):
        h = min(H, len(Zd) - k0)
        y_pred[k0:k0 + h] = rollout(A, B, C, D, Zd[k0], Ud[k0:k0 + h], latent) + d["y_mean"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, y, color="#3a8fb7", lw=0.9, label="real")
    ax.plot(t, y_pred, color="#cf545a", lw=0.6, ls="--", alpha=0.85, label="DMDc rollout 3h")
    for k in [k_train, k_val]:
        ax.axvline(t[k], color="gray", ls=":", lw=1.2)
    ax.set_ylabel("°C")
    ax.set_title("Historia completa: real vs modelo de espacio de estados latente")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "resultado_historia_completa.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t[k_val:], y[k_val:], color="#3a8fb7", lw=1.3, label="real")
    ax.plot(t[k_val:], y_pred[k_val:], color="#cf545a", lw=1.0, ls="--", label="DMDc rollout 3h")
    ax.set_ylabel("°C")
    ax.set_title("Test: real vs modelo")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "resultado_rollout_test.png"), dpi=120)

    eig = np.linalg.eigvals(A)
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
    ax.scatter(eig.real, eig.imag, s=60, color="#cf545a", zorder=5, label="polos DMDc")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Polos en el plano z")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "resultado_polos.png"), dpi=120)

    j = list(m["inputs"]).index("heater_power_pct")
    scale_h = m["u_scale"][j]
    num_z, den_z = signal.ss2tf(A, B[:, [j]] / scale_h, C, D[:, [j]] / scale_h)
    num_z = num_z.ravel()
    w_nyq = np.pi / dt
    w = np.logspace(-6, np.log10(w_nyq), 400)
    w_s = np.logspace(-6, np.log10(w_nyq) + 0.7, 500)
    _, mag_z, _ = signal.dbode((num_z, den_z, dt), w=w * dt)
    _, mag_s, _ = signal.bode((modelo_control.NUM_S, modelo_control.DEN_S), w=w_s)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogx(w, mag_z - mag_z[0], color="#cf545a", lw=1.6, label="G(z) latente")
    ax.semilogx(w_s, mag_s - mag_s[0], color="#3a8fb7", lw=1.6, label="G(s)")
    ax.axvline(w_nyq, color="gray", ls="--", lw=1.0)
    ymin, ymax = ax.get_ylim()
    ax.text(w_nyq * 0.85, ymin + 1, "frecuencia de Nyquist", fontsize=8,
            color="dimgray", rotation=90, va="bottom", ha="right")
    ax.set_xlim(1e-6, w_nyq * 5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("w (rad/s)")
    ax.set_ylabel("|G| normalizada a DC (dB)")
    ax.set_title("Diagrama de Bode (magnitud): heater -> temperatura")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "resultado_bode_gz_gs.png"), dpi=120)

    n_steps = 145
    t_h = np.arange(n_steps) * 5 / 60
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for jj, name in enumerate(m["inputs"]):
        u_step = np.zeros((n_steps, len(m["inputs"])))
        u_step[:, jj] = 1.0 / m["u_scale"][jj]
        y_step = rollout(A, B, C, D, np.zeros(latent), u_step, latent)
        ax.plot(t_h, y_step, lw=1.5, label=str(name))
    ax.set_xlabel("horas")
    ax.set_ylabel("delta temperatura (°C)")
    ax.set_title("Respuesta a escalon de +1 unidad fisica")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "resultado_escalon_entradas.png"), dpi=120)

    print(f"figuras guardadas en {fig_dir}")


if __name__ == "__main__":
    main()
