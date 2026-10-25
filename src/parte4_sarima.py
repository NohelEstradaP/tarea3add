"""Parte 4: del AR(1) al SARIMA, sobre los días abiertos del entrenamiento."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from parte3_descomp import grafica_acf
from utils import FIN_TRAIN, acf, guardar_tabla, ljung_box, serie_diaria


def dummies_dia(idx):
    """Matriz de indicadoras de día de la semana (una columna por día abierto: lun-vie y dom)."""
    dow = pd.Index(idx).day_of_week
    dias = [0, 1, 2, 3, 4, 6]
    return np.column_stack([(dow == d).astype(float) for d in dias])


def ajustar_s1(y, maxiter=50):
    m = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(disp=False, maxiter=maxiter)


def ajustar_s2(y, maxiter=50):
    m = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 6), trend="n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(disp=False, maxiter=maxiter)


def tabla_q(res, gl_menos, nombre):
    filas = []
    for m in (6, 12):
        q, p = ljung_box(res, m, gl=m - gl_menos)
        filas.append({"modelo": nombre, "m": m, "Q(m)": q, "gl": m - gl_menos, "valor p": p})
    return filas


def main():
    s = serie_diaria()
    tr = s[:FIN_TRAIN].dropna()
    y = tr.values

    # 4.1 OLS contra indicadoras de día (sin intercepto)
    X = dummies_dia(tr.index)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    print("medias por día (miles):", np.round(beta / 1000, 2).tolist())
    re = acf(e, 21)
    print("ACF residuos dummies:", np.round(re, 3).tolist())
    grafica_acf(re, len(e), "ACF de los residuos de la regresión con días de la semana", "p4_acf_dummies.png")
    filas = tabla_q(e, 0, "residuos dummies")

    # 4.2 AR(1) sobre los residuos
    Z = np.column_stack([np.ones(len(e) - 1), e[:-1]])
    (c_hat, phi_hat), *_ = np.linalg.lstsq(Z, e[1:], rcond=None)
    u = e[1:] - c_hat - phi_hat * e[:-1]
    print(f"phi_hat = {phi_hat:.4f}, c_hat = {c_hat:.2f}")
    ru = acf(u, 21)
    banda = 1.96 / np.sqrt(len(u))
    print("rezagos fuera AR(1):", (np.where(np.abs(ru) > banda)[0] + 1).tolist())
    grafica_acf(ru, len(u), r"ACF de los residuos del AR(1) sobre $e_t$", "p4_acf_ar1.png")
    filas += tabla_q(u, 1, "AR(1) sobre e_t")

    # 4.3 SARIMA sobre el logaritmo (arreglo de valores, no la serie con fechas)
    print("mínimo de la venta diaria en entrenamiento:", y.min())
    ly = np.log(tr.values)
    params = {}
    for nombre, f in (("S1", ajustar_s1), ("S2", ajustar_s2)):
        r50 = f(ly)
        r500 = f(ly, maxiter=500)
        for etiqueta, r in (("maxiter=50", r50), ("maxiter=500", r500)):
            fila = {"modelo": nombre, "ajuste": etiqueta,
                    "convergió": bool(r.mle_retvals["converged"]), "iteraciones": r.mle_retvals.get("iterations", np.nan),
                    "AIC": r.aic}
            fila.update({k: v for k, v in zip(r.model.param_names, r.params)})
            params[(nombre, etiqueta)] = fila
        print(nombre, "convergió con 50:", r50.mle_retvals["converged"], "| con 500:", r500.mle_retvals["converged"])
    tp = pd.DataFrame(params.values()).set_index(["modelo", "ajuste"])
    tp["convergió"] = tp["convergió"].map({True: "sí", False: "no"})
    guardar_tabla(tp, "p4_sarima_parametros", decimales=4)
    print(tp.to_string())

    # 4.4 diagnóstico con el ajuste que convergió
    s1 = ajustar_s1(ly, 500)
    s2 = ajustar_s2(ly, 500)
    filas += tabla_q(s1.resid, 2, "S1")
    filas += tabla_q(s2.resid[7:], 2, "S2 (sin 7 primeros)")
    grafica_acf(acf(s1.resid, 21), len(s1.resid), "ACF de residuos de S1", "p4_acf_s1.png")
    grafica_acf(acf(s2.resid[7:], 21), len(s2.resid) - 7, "ACF de residuos de S2", "p4_acf_s2.png")
    tq = pd.DataFrame(filas).set_index(["modelo", "m"])
    guardar_tabla(tq, "p4_ljungbox", decimales=4)
    print(tq.to_string())

    # 4.5 pronósticos a horizonte largo
    h = 60
    f1 = np.exp(s1.forecast(h))
    f2 = np.exp(s2.forecast(h))
    media_s1 = np.exp(s1.params[0] / ((1 - s1.params[1]) * (1 - s1.params[2])))
    print("S1 h=1,6,30,60:", np.round(f1[[0, 5, 29, 59]], 0), "exp(media incondicional):", round(media_s1))
    print("S2 h=1..12:", np.round(f2[:12], 0))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils import FIG
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.arange(-60, 0), tr.values[-60:] / 1000, c="black", lw=0.8, label="últimos 60 días abiertos")
    ax.plot(np.arange(h), f1 / 1000, label="S1")
    ax.plot(np.arange(h), f2 / 1000, label="S2")
    ax.set_xlabel("horizonte (días abiertos)")
    ax.set_ylabel("miles de £")
    ax.legend()
    ax.set_title("Pronóstico desde el 30 de septiembre a 60 días abiertos")
    fig.tight_layout()
    fig.savefig(FIG / "p4_horizonte.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
