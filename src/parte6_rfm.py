"""Parte 6: análisis RFM de los clientes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import FIG, guardar_tabla, datos_limpios

CORTE = pd.Timestamp("2011-10-01")
FIN = pd.Timestamp("2011-12-09")  # exclusivo: hasta el 8 de diciembre inclusive
SEMILLA_RFM = 2026


def rfm(df, corte=CORTE):
    """R, F y M por cliente usando solo transacciones anteriores a corte."""
    antes = df[df["InvoiceDate"] < corte]
    ventas = antes[~antes["InvoiceNo"].str.startswith("C")]
    tabla = pd.DataFrame({
        "R": (corte - ventas.groupby("CustomerID")["InvoiceDate"].max()).dt.days,
        "F": ventas.groupby("CustomerID")["InvoiceNo"].nunique(),
        "M": antes.groupby("CustomerID")["Monto"].sum(),
    })
    return tabla.sort_index()


def puntajes(t, semilla=SEMILLA_RFM):
    """Quintiles 1-5. Empates al azar: orden por CustomerID, permutación con la semilla y rank 'first'."""
    t = t.sort_index()
    perm = np.random.default_rng(semilla).permutation(len(t))
    b = t.iloc[perm]
    sc = pd.DataFrame(index=b.index)
    sc["R_s"] = pd.qcut((-b["R"]).rank(method="first"), 5, labels=range(1, 6)).astype(int)
    sc["F_s"] = pd.qcut(b["F"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
    sc["M_s"] = pd.qcut(b["M"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
    return sc.sort_index()


def main():
    df = datos_limpios(con_cliente=True)
    t = rfm(df)
    antes = df[df["InvoiceDate"] < CORTE]
    solo_canc = set(antes["CustomerID"]) - set(t.dropna().index)
    t = t.dropna()
    print(f"clientes con al menos una compra antes del corte: {len(t)}; solo cancelaciones: {len(solo_canc)}")
    m0 = t[t["M"] <= 0]
    print(f"M <= 0: {len(m0)}"); print(m0.to_string())
    print(t.describe().round(1).to_string())

    # 6.2 puntajes
    sc = puntajes(t)
    t = t.join(sc)
    print("F: proporción de clientes con F=1:", (t["F"] == 1).mean().round(3),
          "| valores únicos de F:", t["F"].nunique())
    rango_f = t.groupby("F_s")["F"].agg(["min", "max", "size"])
    rango_f.index.name = "puntaje F"
    guardar_tabla(rango_f, "p6_rangos_f")
    print(rango_f.to_string())

    # 6.3 ¿quién compró del 1 de octubre al 8 de diciembre?
    desp = df[(df["InvoiceDate"] >= CORTE) & (df["InvoiceDate"] < FIN)]
    compradores = set(desp.loc[~desp["InvoiceNo"].str.startswith("C"), "CustomerID"])
    t["compra"] = t.index.isin(compradores).astype(int)
    frac = pd.DataFrame({d: t.groupby(f"{d}_s")["compra"].mean() for d in "RFM"})
    frac.index.name = "puntaje"
    guardar_tabla(frac, "p6_fraccion_compra", decimales=3)
    print(frac.round(3).to_string())
    t["suma"] = t["R_s"] + t["F_s"] + t["M_s"]
    auc = pd.DataFrame({"AUC": {"R (invertida)": roc_auc_score(t["compra"], -t["R"]),
                                "F": roc_auc_score(t["compra"], t["F"]),
                                "M": roc_auc_score(t["compra"], t["M"]),
                                "R_s + F_s + M_s": roc_auc_score(t["compra"], t["suma"])}})
    guardar_tabla(auc, "p6_auc", decimales=3)
    print("tasa base de compra:", t["compra"].mean().round(3)); print(auc.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for d in "RFM":
        ax.plot(frac.index, frac[d], marker="o", label=d)
    ax.axhline(t["compra"].mean(), ls="--", c="gray", label="tasa base")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("puntaje")
    ax.set_ylabel("fracción que compró oct-dic")
    ax.legend()
    ax.set_title("Fracción de clientes que compró por puntaje RFM")
    fig.tight_layout()
    fig.savefig(FIG / "p6_fraccion_compra.png", dpi=150)
    plt.close(fig)

    # 6.4 campeones y 6.5 clientes nuevos
    venta_cli = desp.groupby("CustomerID")["Monto"].sum()
    total = venta_cli.sum()
    campeones = t[(t["R_s"] >= 4) & (t["F_s"] >= 4) & (t["M_s"] >= 4)].index
    # nuevo = sin ninguna factura de venta antes del corte (incluye a los que solo tenían cancelaciones)
    nuevos = venta_cli.index.difference(t.index)
    resumen = pd.DataFrame({"valor": [
        len(t), len(campeones), len(campeones) / len(t),
        venta_cli.reindex(campeones).fillna(0).sum() / total,
        len(nuevos), venta_cli[nuevos].sum() / total, total]},
        index=["clientes RFM", "campeones", "fracción de clientes campeones",
               "fracción de la venta oct-dic de campeones",
               "clientes nuevos que compraron oct-dic", "fracción de la venta oct-dic de nuevos",
               "venta neta oct-dic con CustomerID (£)"])
    guardar_tabla(resumen, "p6_campeones_nuevos", decimales=3)
    print(resumen.to_string())


if __name__ == "__main__":
    main()
