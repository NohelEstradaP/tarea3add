"""Parte 0: procesos simulados (AR(1), caminata aleatoria y procesos sin memoria)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIG, acf, guardar_tabla, ljung_box, ols_phi

N = 300
BURN = 200
REPS = 5000
PHIS = [0, 0.5, 0.9, 1]
SEMILLA = 14092  # cada escenario usa SEMILLA + i, así todos son distintos


def simular_ar1(phi, n=N, reps=1, c=0.0, sigma=1.0, burn=BURN, rng=None):
    """Simula reps series de un AR(1) con Y_0 = 0 y descarta los primeros burn pasos."""
    rng = np.random.default_rng() if rng is None else rng
    eps = rng.normal(0, sigma, size=(reps, n + burn))
    y = np.zeros((reps, n + burn))
    prev = np.zeros(reps)
    for t in range(n + burn):
        prev = c + phi * prev + eps[:, t]
        y[:, t] = prev
    return y[:, burn:]


def sin_memoria(tipo, reps, rng):
    t = np.arange(1, N + 1)
    if tipo == "tendencia":
        m = 2 * t / N
    else:
        m = np.where(t % 7 == 6, -1.5, 0.25)
    return m + rng.normal(0, 1, size=(reps, N))


def resumen(y, nombre, extra=None):
    r = acf(y, 7)
    banda = 1.96 / np.sqrt(N)
    phi_hat = ols_phi(y)
    rech = (np.abs(r[:, 0]) > banda).astype(float)
    fila = {
        "proceso": nombre,
        "r1": r[:, 0].mean(), "ee(r1)": r[:, 0].std(ddof=1) / np.sqrt(len(r)),
        "r5": r[:, 4].mean(), "ee(r5)": r[:, 4].std(ddof=1) / np.sqrt(len(r)),
        "rechazo r1": rech.mean(), "ee(rech)": rech.std(ddof=1) / np.sqrt(len(r)),
        "r7": r[:, 6].mean(), "ee(r7)": r[:, 6].std(ddof=1) / np.sqrt(len(r)),
        "phi OLS": phi_hat.mean(), "ee(phi)": phi_hat.std(ddof=1) / np.sqrt(len(r)),
    }
    if extra:
        fila.update(extra)
    return fila


def main():
    # 0.2: una realización de cada phi
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for i, phi in enumerate(PHIS):
        y = simular_ar1(phi, rng=np.random.default_rng(SEMILLA + 100 + i))[0]
        ax[i].plot(y, lw=0.8)
        ax[i].set_title(rf"$\phi = {phi}$", fontsize=10)
    ax[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(FIG / "p0_realizaciones.png", dpi=150)
    plt.close(fig)

    # 0.2: 5000 réplicas por phi
    filas, series = [], {}
    for i, phi in enumerate(PHIS):
        y = simular_ar1(phi, reps=REPS, rng=np.random.default_rng(SEMILLA + i))
        series[f"AR(1) phi={phi}"] = y
        filas.append(resumen(y, f"phi = {phi}", {"phi^5": phi ** 5}))
    tab = pd.DataFrame(filas).set_index("proceso")
    t02 = tab[["r1", "ee(r1)", "r5", "ee(r5)", "phi^5", "rechazo r1", "ee(rech)", "phi OLS", "ee(phi)"]]
    guardar_tabla(t02, "p0_tabla_ar1", decimales=4)
    print(t02.round(4).to_string())

    # 0.4: procesos sin memoria con media que cambia
    filas = [resumen(series["AR(1) phi=0"], "ruido blanco")]
    for j, tipo in enumerate(["tendencia", "semanal"]):
        y = sin_memoria(tipo, REPS, np.random.default_rng(SEMILLA + 10 + j))
        series[tipo] = y
        filas.append(resumen(y, tipo))
    t04 = pd.DataFrame(filas).set_index("proceso")[["r1", "ee(r1)", "rechazo r1", "ee(rech)", "r7", "ee(r7)"]]
    guardar_tabla(t04, "p0_tabla_sin_memoria", decimales=4)
    print(t04.round(4).to_string())

    # figura de un ejemplo de cada proceso sin memoria y su ACF promedio
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for j, tipo in enumerate(["tendencia", "semanal"]):
        ax[j, 0].plot(series[tipo][0], lw=0.8)
        ax[j, 0].set_title(f"Una realización: {tipo}", fontsize=10)
        r = acf(series[tipo], 21).mean(axis=0)
        ax[j, 1].bar(range(1, 22), r)
        ax[j, 1].axhline(1.96 / np.sqrt(N), ls="--", c="gray")
        ax[j, 1].axhline(-1.96 / np.sqrt(N), ls="--", c="gray")
        ax[j, 1].set_title(f"ACF promedio: {tipo}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "p0_sin_memoria.png", dpi=150)
    plt.close(fig)

    # 0.5: Ljung-Box con m = 10 al 5 %
    from statsmodels.stats.diagnostic import acorr_ljungbox
    filas = []
    for nombre, clave in [("ruido blanco", "AR(1) phi=0"), ("tendencia", "tendencia"), ("semanal", "semanal")]:
        y = series[clave]
        q, p = ljung_box(y, 10)
        rech = (p < 0.05).astype(float)
        # verificación contra statsmodels en las primeras 50 réplicas
        p_sm = np.array([acorr_ljungbox(fila, lags=[10])["lb_pvalue"].iloc[0] for fila in y[:50]])
        filas.append({"proceso": nombre, "rechazo LB(10)": rech.mean(),
                      "ee": rech.std(ddof=1) / np.sqrt(len(rech)),
                      "rechazo r1": (np.abs(acf(y, 1)[:, 0]) > 1.96 / np.sqrt(N)).mean(),
                      "max |p - p_sm|": np.abs(p[:50] - p_sm).max()})
    t05 = pd.DataFrame(filas).set_index("proceso")
    guardar_tabla(t05, "p0_tabla_ljungbox", decimales=4)
    print(t05.to_string())


if __name__ == "__main__":
    main()
