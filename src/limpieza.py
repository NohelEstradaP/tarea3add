"""Limpieza de Online Retail. La función limpiar() se reusa en la Parte 6 y en el proyecto."""
import pandas as pd

# Códigos que no son productos: envíos, cargos, comisiones, ajustes y vales de regalo.
# Los DCGS* y PADS sí son productos (tienen descripción de artículo y precio normal).
NO_PRODUCTOS = {"POST", "DOT", "C2", "M", "m", "D", "S", "B", "BANK CHARGES",
                "AMAZONFEE", "CRUK"}
PREFIJO_VALE = "gift_"


def cargar(ruta):
    """Lee el .xlsx original. InvoiceNo, StockCode y CustomerID se leen como texto:
    InvoiceNo tiene prefijos (C, A), StockCode tiene letras (85123A, POST) y
    CustomerID como número se vuelve float (17850.0) por los faltantes."""
    return pd.read_excel(ruta, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str})


def _anulaciones_manuales(df):
    """Ventas anuladas con una línea 'M' (Manual) en vez de una cancelación del producto:
    mismo cliente, mismo día y mismo monto con signo contrario. Devuelve el índice de ambas líneas."""
    monto = df["Quantity"] * df["UnitPrice"]
    dia = df["InvoiceDate"].dt.normalize()
    canc = df[df["InvoiceNo"].str.startswith("C") & (df["StockCode"] == "M")]
    ventas = df[~df["InvoiceNo"].str.startswith("C") & (df["Quantity"] > 0)]
    quitar = []
    for i, f in canc.iterrows():
        cand = ventas[(ventas["CustomerID"] == f["CustomerID"]) & (dia[ventas.index] == dia[i])
                      & ((monto[ventas.index] + monto[i]).abs() < 1e-6)]
        if f["CustomerID"] is not None and pd.notna(f["CustomerID"]) and len(cand):
            quitar += [i, cand.index[0]]
    return pd.Index(quitar)


def limpiar(df, con_cliente=False, bitacora=None):
    """Devuelve el DataFrame limpio con la columna Monto = Quantity * UnitPrice.

    Reglas, en este orden:
      1. duplicados exactos;
      2. facturas con prefijo A (ajustes de deuda incobrable);
      3. ventas anuladas con una línea Manual (se quitan la venta y la anulación);
      4. códigos que no son productos (envíos, cargos, comisiones, vales);
      5. cantidades negativas en facturas que no son cancelaciones (ajustes de inventario);
      6. precios iguales a cero o negativos;
      7. (solo si con_cliente) filas sin CustomerID.
    Las cancelaciones (prefijo C) se quedan: restan el día en que ocurren.
    Si bitacora es una lista, se le agrega un dict por regla con filas y monto eliminados.
    """
    df = df.copy()
    df["Monto"] = df["Quantity"] * df["UnitPrice"]

    def aplicar(nombre, mascara_quitar):
        nonlocal df
        q = df[mascara_quitar]
        if bitacora is not None:
            bitacora.append({"orden": len(bitacora) + 1, "regla": nombre, "filas": len(q),
                             "monto": q["Monto"].sum(), "monto_abs": q["Monto"].abs().sum()})
        df = df[~mascara_quitar]

    aplicar("duplicados exactos", df.duplicated())
    aplicar("factura con prefijo A", df["InvoiceNo"].str.startswith("A"))
    aplicar("venta anulada con Manual", df.index.isin(_anulaciones_manuales(df)))
    aplicar("código que no es producto",
            df["StockCode"].isin(NO_PRODUCTOS) | df["StockCode"].str.startswith(PREFIJO_VALE))
    aplicar("cantidad negativa sin cancelación",
            (df["Quantity"] < 0) & ~df["InvoiceNo"].str.startswith("C"))
    aplicar("precio <= 0", df["UnitPrice"] <= 0)
    if con_cliente:
        aplicar("sin CustomerID", df["CustomerID"].isna())
    return df.reset_index(drop=True)
