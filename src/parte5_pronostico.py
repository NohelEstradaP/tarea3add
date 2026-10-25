"""Parte 5: pronóstico con origen móvil semanal y comparación de validaciones."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from parte3_descomp import media_movil
from parte4_sarima import ajustar_s1, ajustar_s2, dummies_dia
from utils import FIG, FIN_TEST, FIN_TRAIN, guardar_tabla, serie_diaria

SEMILLAS_CV = list(range(10))  # semillas 0..9 para la validación cruzada aleatoria
INICIO = pd.Timestamp("2010-12-01")


# ---------- matrices de diseño ----------

def rezago_semanal(s, fechas, k):
    """y_{t-k} con k = 7 o 14: el valor del mismo día de la semana abierto más reciente
    en o antes de t-k. s es la serie con calendario completo (cerrados = NaN)."""
    obs = s.dropna()
    out = []
    for t in fechas:
        d = t - pd.Timedelta(days=k)
        while d >= obs.index[0] and d not in obs.index:
            d -= pd.Timedelta(days=7)
        out.append(obs[d] if d in obs.index else np.nan)
    return np.array(out)


def X_m1(fechas):
    t = (pd.Index(fechas) - INICIO).days.values.astype(float)
    return np.column_stack([t, dummies_dia(fechas)])


def X_m2(s, fechas, extra=None):
    X = np.column_stack([dummies_dia(fechas), rezago_semanal(s, fechas, 7), rezago_semanal(s, fechas, 14)])
    if extra is not None:
        X = np.column_stack([X, extra.reindex(fechas).values])
    return X


def ols(X, y):
    ok = ~np.isnan(X).any(axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


# ---------- modelos: historia (serie con calendario hasta el origen) -> pronóstico de objetivos ----------

def pron_b1(hist, objetivos, s):
    return np.full(len(objetivos), hist.dropna().mean())


def pron_b2(hist, objetivos, s):
    obs = hist.dropna()
    return np.array([obs[obs.index.day_of_week == t.day_of_week].iloc[-1] for t in objetivos])


def pron_m1(hist, objetivos, s):
    obs = hist.dropna()
    beta = ols(X_m1(obs.index), obs.values)
    return X_m1(objetivos) @ beta


def pron_m2(hist, objetivos, s, extra=None):
    obs = hist.dropna()
    beta = ols(X_m2(hist, obs.index, extra), obs.values)
    # los rezagos de los objetivos se leen de la historia: t-7 <= origen siempre
    return X_m2(hist, objetivos, extra) @ beta


CONVERGENCIA = []


def _pron_sarima(ajustar, maxiter, nombre):
    def f(hist, objetivos, s):
        obs = hist.dropna()
        r = ajustar(np.log(obs.values), maxiter=maxiter)
        conv = bool(r.mle_retvals["converged"])
        if not conv:
            # L-BFGS se atasca en la búsqueda de línea; se reajusta con Nelder-Mead desde ese punto
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = r.model.fit(start_params=r.params, method="nm", maxiter=5000, disp=False)
        CONVERGENCIA.append({"modelo": nombre, "origen": hist.index[-1], "convergió L-BFGS": conv,
                             "convergió final": bool(r.mle_retvals["converged"])})
        # horizonte = posición del objetivo entre los días abiertos posteriores al origen
        return np.exp(r.forecast(len(objetivos)))
    return f


MODELOS = {"B1": pron_b1, "B2": pron_b2, "M1": pron_m1, "M2": pron_m2,
           "S1": _pron_sarima(ajustar_s1, 500, "S1"), "S2": _pron_sarima(ajustar_s2, 50, "S2")}


# ---------- esquemas de validación ----------

def origen_movil(s, primer_origen, ultimo_origen, fin_eval, modelo, extra=None):
    """Cada domingo pronostica los días abiertos de lunes a domingo siguientes (hasta fin_eval)."""
    reales, preds, fechas = [], [], []
    for o in pd.date_range(primer_origen, ultimo_origen, freq="W-SUN"):
        hist = s[:o]
        sem = s[o + pd.Timedelta(days=1): min(o + pd.Timedelta(days=7), fin_eval)].dropna()
        if len(sem) == 0:
            continue
        if extra is None:
            p = MODELOS[modelo](hist, sem.index, s)
        else:
            p = pron_m2(hist, sem.index, s, extra=extra)
        reales.append(sem.values)
        preds.append(p)
        fechas.append(sem.index)
    return (np.concatenate(reales), np.concatenate(preds), np.concatenate([f.values for f in fechas]))


def metricas(y, p):
    e = y - p
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))


def cv_aleatoria(s, modelo, extra=None):
    """5 folds aleatorios sobre los días abiertos del entrenamiento; una corrida por semilla."""
    tr = s[:FIN_TRAIN].dropna()
    if modelo == "B1":
        X = np.ones((len(tr), 1))
    elif modelo == "M1":
        X = X_m1(tr.index)
    else:
        X = X_m2(s, tr.index, extra)
    ok = ~np.isnan(X).any(axis=1)
    X, y = X[ok], tr.values[ok]
    res = []
    for sem in SEMILLAS_CV:
        e = np.empty(len(y))
        for a, b in KFold(5, shuffle=True, random_state=sem).split(X):
            beta, *_ = np.linalg.lstsq(X[a], y[a], rcond=None)
            e[b] = y[b] - X[b] @ beta
        res.append((np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))))
    res = np.array(res)
    return res.mean(axis=0), res.std(axis=0, ddof=1)


def main():
    s = serie_diaria()
    filas = {}
    pred_test = {}
    for m in MODELOS:
        fila = {}
        if m in ("B1", "M1", "M2"):
            (mae, rmse), (sd_mae, sd_rmse) = cv_aleatoria(s, m)
            fila.update({"MAE (a)": mae, "sd MAE (a)": sd_mae, "RMSE (a)": rmse, "sd RMSE (a)": sd_rmse})
        y, p, _ = origen_movil(s, "2011-05-01", "2011-09-25", FIN_TRAIN, m)
        fila["MAE (b)"], fila["RMSE (b)"] = metricas(y, p)
        y, p, f = origen_movil(s, "2011-10-02", "2011-12-04", FIN_TEST, m)
        fila["MAE (c)"], fila["RMSE (c)"] = metricas(y, p)
        pred_test[m] = pd.Series(p, index=pd.DatetimeIndex(f))
        filas[m] = fila
        print(m, {k: round(v) for k, v in fila.items()})
    tab = pd.DataFrame(filas).T
    tab = tab[["MAE (a)", "sd MAE (a)", "MAE (b)", "MAE (c)", "RMSE (a)", "sd RMSE (a)", "RMSE (b)", "RMSE (c)"]]
    guardar_tabla(tab / 1000, "p5_validacion", decimales=2)
    conv = pd.DataFrame(CONVERGENCIA)
    print(conv.groupby("modelo")[["convergió L-BFGS", "convergió final"]].agg(["sum", "size"]).to_string())
    print("orígenes donde L-BFGS no convergió:\n", conv[~conv["convergió L-BFGS"]].to_string())

    # 5.3 gráfica del periodo de prueba
    real = s["2011-10-01":FIN_TEST]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, grupo in zip(axes, (["B1", "B2", "M1"], ["M2", "S1", "S2"])):
        ax.plot(real.index, real / 1000, c="black", marker=".", lw=1, label="real")
        for m in grupo:
            ax.plot(pred_test[m].index, pred_test[m] / 1000, marker=".", lw=0.9, label=m)
        ax.set_ylabel("miles de £")
        ax.legend(ncol=4)
    axes[0].set_title("Pronósticos con origen móvil semanal en el periodo de prueba")
    fig.tight_layout()
    fig.savefig(FIG / "p5_prueba.png", dpi=150)
    plt.close(fig)

    # error por semana en prueba (para ver dónde gana cada modelo)
    err = pd.DataFrame({m: (pred_test[m] - s.reindex(pred_test[m].index)).abs() for m in MODELOS})
    sem = err.groupby(err.index.to_period("W-SUN")).mean() / 1000
    sem.index = sem.index.start_time.strftime("%m-%d")
    guardar_tabla(sem, "p5_mae_semanal", decimales=1)

    # 5.4 la trampa: M2 + media móvil centrada calculada una vez sobre toda la serie
    ma = media_movil(s)
    (mae, rmse), (sd_mae, sd_rmse) = cv_aleatoria(s, "M2", extra=ma)
    y, p, _ = origen_movil(s, "2011-05-01", "2011-09-25", FIN_TRAIN, "M2", extra=ma)
    mae_b, rmse_b = metricas(y, p)
    trampa = pd.DataFrame({"MAE (a)": [tab.loc["M2", "MAE (a)"], mae], "RMSE (a)": [tab.loc["M2", "RMSE (a)"], rmse],
                           "MAE (b)": [tab.loc["M2", "MAE (b)"], mae_b], "RMSE (b)": [tab.loc["M2", "RMSE (b)"], rmse_b]},
                          index=["M2", "M2 + MA7 centrada"])
    guardar_tabla(trampa / 1000, "p5_trampa", decimales=2)
    print(trampa.round(0).to_string())


if __name__ == "__main__":
    main()
