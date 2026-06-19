import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from preprocesamiento import cargar_config

A = np.array([[0.0, 1.0], [-4.44e-6, -7.33e-3]])
B = np.array([[0.0], [1.0]])
C = np.array([[3.11e-3, -0.4667]])
K = np.array([[-4.0e-6, -6.0e-3]])
N = 1.43e-4
NUM_S = [-0.4667, 3.11e-3]
DEN_S = [1.0, 7.33e-3, 4.44e-6]


def main():
    cfg = cargar_config()
    poles = np.linalg.eigvals(A)
    tau = -1 / poles.real / 60
    dc = (C @ np.linalg.solve(-A, B)).item()
    cero = NUM_S[1] / -NUM_S[0]
    print(f"polos (1/s): {poles}")
    print(f"constantes de tiempo: {tau.round(1)} min")
    print(f"ganancia DC G(0): {dc:.1f}")
    print(f"cero: s = {cero:+.2e}")

    poles_lc = np.linalg.eigvals(A - B @ K)
    print(f"polos lazo cerrado (1/s): {poles_lc}")

    t = np.linspace(0, 6 * 3600, 2000)
    _, y_step = signal.step(signal.StateSpace(A, B, C, np.zeros((1, 1))), T=t)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t / 3600, y_step, color="#3a8fb7", lw=1.6)
    ax.set_xlabel("horas")
    ax.set_ylabel("respuesta")
    ax.set_title("G(s): respuesta a escalon unitario")
    plt.tight_layout()
    out = os.path.join(cfg["figuras"], "control_escalon_gs.png")
    plt.savefig(out, dpi=120)
    print(f"guardado: {out}")


if __name__ == "__main__":
    main()
