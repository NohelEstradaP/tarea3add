"""Funciones compartidas: rutas, autocorrelación a mano, Ljung-Box y tablas."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Online Retail.xlsx"
FIG = ROOT / "figuras"
TAB = ROOT / "tablas"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)


def acf(y, kmax):
    """r_k para k = 1..kmax. Si y es 2D, calcula por fila (cada fila es una serie)."""
    y = np.asarray(y, dtype=float)
    unidim = y.ndim == 1
    if unidim:
        y = y[None, :]
    n = y.shape[1]
    d = y - y.mean(axis=1, keepdims=True)
    den = (d ** 2).sum(axis=1)
    r = np.empty((y.shape[0], kmax))
    for k in range(1, kmax + 1):
        r[:, k - 1] = (d[:, k:] * d[:, : n - k]).sum(axis=1) / den
    return r[0] if unidim else r


def ljung_box(y, m, gl=None):
    """Q(m) de Ljung-Box y su valor p. gl = grados de libertad (m por defecto)."""
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    r = acf(y, m)
    k = np.arange(1, m + 1)
    q = n * (n + 2) * (r ** 2 / (n - k)).sum(axis=-1)
    gl = m if gl is None else gl
    return q, stats.chi2.sf(q, gl)


def ols_phi(y):
    """Estimado OLS de phi en y_t = c + phi*y_{t-1} + e_t, por fila."""
    y = np.asarray(y, dtype=float)
    x, z = y[..., :-1], y[..., 1:]
    xc = x - x.mean(axis=-1, keepdims=True)
    zc = z - z.mean(axis=-1, keepdims=True)
    return (xc * zc).sum(axis=-1) / (xc ** 2).sum(axis=-1)


def guardar_tabla(df, nombre, decimales=3, index=True):
    """Guarda la tabla como .csv y como .tex (tabular simple con booktabs)."""
    df.to_csv(TAB / f"{nombre}.csv", index=index)
    d = df.reset_index() if index else df.copy()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "--"
            return f"{v:,.{decimales}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    lineas = [r"\begin{tabular}{" + "l" + "r" * (d.shape[1] - 1) + "}", r"\toprule",
              " & ".join(esc(c) for c in d.columns) + r" \\", r"\midrule"]
    for _, fila in d.iterrows():
        lineas.append(" & ".join(esc(fmt(v)) for v in fila) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{nombre}.tex").write_text("\n".join(lineas) + "\n")
    return df
