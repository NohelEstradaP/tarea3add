"""Parte 1: exploración y limpieza."""
import pandas as pd

from limpieza import cargar, limpiar
from utils import DATA, guardar_tabla


def main():
    df = cargar(DATA)
    df["Monto"] = df["Quantity"] * df["UnitPrice"]

    # 1.1 conteos básicos
    conteos = pd.DataFrame({"valor": [len(df), df["InvoiceNo"].nunique(), df["StockCode"].nunique(),
                                      df["CustomerID"].nunique(), df["CustomerID"].isna().mean()]},
                           index=["filas", "facturas", "productos (StockCode)", "clientes",
                                  "fracción sin CustomerID"])
    guardar_tabla(conteos, "p1_conteos", decimales=4)
    print(conteos.to_string())

    # 1.2 cada caso por separado (sobre los datos crudos)
    es_c = df["InvoiceNo"].str.startswith("C")
    casos = {
        "(a) duplicados exactos": df.drop(columns="Monto").duplicated(),
        "(b) prefijo C": es_c,
        "(b) prefijo A": df["InvoiceNo"].str.startswith("A"),
        "(c) StockCode sin 5 dígitos": ~df["StockCode"].str[:5].str.isdigit(),
        "(d) precio = 0": df["UnitPrice"] == 0,
        "(d) precio < 0": df["UnitPrice"] < 0,
        "(e) cantidad < 0 sin C": (df["Quantity"] < 0) & ~es_c,
        "(f) sin CustomerID": df["CustomerID"].isna(),
    }
    t12 = pd.DataFrame({"filas": {k: int(v.sum()) for k, v in casos.items()},
                        "monto": {k: df.loc[v, "Monto"].sum() for k, v in casos.items()}})
    guardar_tabla(t12, "p1_casos", decimales=2)
    print(t12.to_string())

    codigos = (df[casos["(c) StockCode sin 5 dígitos"]]
               .groupby("StockCode")
               .agg(filas=("Quantity", "size"), descripcion=("Description", "first"),
                    precio_medio=("UnitPrice", "mean"), monto=("Monto", "sum"))
               .sort_values("filas", ascending=False))
    codigos["descripcion"] = codigos["descripcion"].fillna("(vacía)").str.slice(0, 32)
    guardar_tabla(codigos, "p1_codigos_no_numericos", decimales=2)

    neg = df[casos["(e) cantidad < 0 sin C"]]
    desc = neg["Description"].fillna("(vacía)").str.lower().value_counts().head(12).to_frame("filas")
    guardar_tabla(desc, "p1_negativos_sin_c")
    print("precio de esas filas:", neg["UnitPrice"].unique())

    # 1.3 bitácora
    bit = []
    limpio = limpiar(df.drop(columns="Monto"), bitacora=bit)
    bit = pd.DataFrame(bit).set_index("orden")
    bit.loc["total"] = ["", bit["filas"].sum(), bit["monto"].sum(), bit["monto_abs"].sum()]
    guardar_tabla(bit, "p1_bitacora", decimales=2)
    print(bit.to_string())
    print("filas limpias:", len(limpio), "venta neta:", limpio["Monto"].sum())

    # 1.4 las seis líneas con mayor |Q x P| antes de limpiar
    top = df.reindex(df["Monto"].abs().sort_values(ascending=False).index).head(6)
    top = top[["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "Monto",
               "InvoiceDate", "CustomerID"]].copy()
    top["sigue en limpio"] = ["sí" if ((limpio["InvoiceNo"] == r.InvoiceNo) & (limpio["StockCode"] == r.StockCode)).any()
                              else "no" for r in top.itertuples()]
    top["InvoiceDate"] = top["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
    top["Description"] = top["Description"].str.slice(0, 26)
    guardar_tabla(top, "p1_top6", decimales=2, index=False)
    print(top.to_string())
    for r in top.itertuples():
        dia = pd.Timestamp(r.InvoiceDate).normalize()
        neto = limpio.loc[limpio["InvoiceDate"].dt.normalize() == dia, "Monto"].sum()
        print(r.InvoiceNo, "venta neta limpia del día", dia.date(), round(neto, 2))


if __name__ == "__main__":
    main()
