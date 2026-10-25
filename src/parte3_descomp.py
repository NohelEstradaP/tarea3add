"""Parte 3: tendencia, estacionalidad y autocorrelación."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIG, FIN_TRAIN, acf, guardar_tabla, ljung_box, serie_diaria

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def grafica_acf(r, n, titulo, archivo):
    banda = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    k = np.arange(1, len(r) + 1)
    ax.bar(k, r, color=np.where(np.abs(r) > banda, "C3", "C0"))
    ax.axhline(banda, ls="--", c="gray")
    ax.axhline(-banda, ls="--", c="gray")
    ax.axhline(0, c="black", lw=0.6)
    ax.set_xticks(k)
    ax.set_xlabel("rezago k (días abiertos)")
    ax.set_ylabel("$r_k$")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(FIG / archivo, dpi=150)
    plt.close(fig)


def media_movil(s):
    """Media móvil centrada de 7 días sobre el calendario completo, con cerrados = 0."""
    return s.fillna(0).rolling(7, center=True).mean()


def indices_estacionales(s):
    ma = media_movil(s)
    tr = s[:FIN_TRAIN].dropna()
    cociente = tr / ma[tr.index]
    idx = cociente.groupby(cociente.index.day_of_week).mean()
    idx.index = [DIAS[i] for i in idx.index]
    return idx


def main():
    s = serie_diaria()
    ma = media_movil(s)

    # 3.1
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, s / 1000, lw=0.7, alpha=0.6, label="venta neta diaria")
    ax.plot(ma.index, ma / 1000, lw=2, c="C3", label="media móvil centrada de 7 días")
    ax.set_ylabel("miles de £")
    ax.legend()
    ax.set_title("Serie diaria y tendencia")
    fig.tight_layout()
    fig.savefig(FIG / "p3_tendencia.png", dpi=150)
    plt.close(fig)
    mensual = ma.resample("MS").mean()
    print("MA7 promedio por mes:\n", (mensual / 1000).round(1).to_string())

    # 3.2
    idx = indices_estacionales(s)
    t32 = idx.to_frame("índice estacional")
    t32.loc["promedio"] = idx.mean()
    guardar_tabla(t32, "p3_indices", decimales=3)
    print(t32.to_string())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(idx.index, idx.values)
    ax.axhline(1, ls="--", c="gray")
    ax.set_ylabel("$y_t$ / MA$_t$ promedio")
    ax.set_title("Índice estacional por día de la semana (entrenamiento)")
    fig.tight_layout()
    fig.savefig(FIG / "p3_indices.png", dpi=150)
    plt.close(fig)

    # 3.3 ACF sobre días abiertos del entrenamiento
    y = s[:FIN_TRAIN].dropna().values
    n = len(y)
    r = acf(y, 21)
    banda = 1.96 / np.sqrt(n)
    fuera = np.where(np.abs(r) > banda)[0] + 1
    print(f"n = {n}, banda = {banda:.3f}, rezagos fuera: {fuera.tolist()} ({len(fuera)})")
    print("r_k:", np.round(r, 3).tolist())
    pico = int(np.argmax(r)) + 1
    print("pico en k =", pico)
    grafica_acf(r, n, "ACF de la venta diaria (días abiertos del entrenamiento)", "p3_acf.png")
    filas = []
    for m in (6, 12):
        q, p = ljung_box(y, m)
        filas.append({"m": m, "Q(m)": q, "gl": m, "valor p": p})
    tq = pd.DataFrame(filas).set_index("m")
    guardar_tabla(tq, "p3_ljungbox", decimales=4)
    print(tq.to_string())
    pd.DataFrame({"k": np.arange(1, 22), "r_k": r}).to_csv("../tablas/p3_acf.csv", index=False)


if __name__ == "__main__":
    main()
