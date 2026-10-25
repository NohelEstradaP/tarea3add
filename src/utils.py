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
            if float(v).is_integer() and abs(v) >= 10:
                return f"{v:,.0f}"
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


# Partición de la Parte 2 en adelante
FIN_TRAIN = pd.Timestamp("2011-09-30")
INICIO_TEST = pd.Timestamp("2011-10-03")
FIN_TEST = pd.Timestamp("2011-12-08")


def datos_limpios(con_cliente=False):
    """Carga el .xlsx, aplica limpiar() y guarda una copia en data/ para no releer el Excel."""
    from limpieza import cargar, limpiar
    cache = ROOT / "data" / ("limpio_cliente.pkl" if con_cliente else "limpio.pkl")
    if cache.exists() and cache.stat().st_mtime > DATA.stat().st_mtime:
        return pd.read_pickle(cache)
    df = limpiar(cargar(DATA), con_cliente=con_cliente)
    df.to_pickle(cache)
    return df


def serie_diaria(df=None, incluir_ultimo=False):
    """Venta neta diaria sobre el calendario completo. Los días sin transacciones quedan NaN."""
    df = datos_limpios() if df is None else df
    s = df.groupby(df["InvoiceDate"].dt.normalize())["Monto"].sum()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    s.index.name = "fecha"
    s.name = "venta"
    return s if incluir_ultimo else s.iloc[:-1]


def dias_abiertos(s):
    """La serie solo con días abiertos (una fila por día, sin calendario)."""
    return s.dropna()
