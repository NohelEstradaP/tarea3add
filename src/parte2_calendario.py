"""Parte 2: la serie diaria y su calendario."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import FIG, FIN_TEST, FIN_TRAIN, INICIO_TEST, datos_limpios, guardar_tabla, serie_diaria

# Bank holidays de Inglaterra y Gales (gov.uk) en el periodo de los datos
FERIADOS = {
    "2010-12-27": "Christmas Day (sustituto)",
    "2010-12-28": "Boxing Day (sustituto)",
    "2011-01-03": "New Year's Day (sustituto)",
    "2011-04-22": "Good Friday (móvil)",
    "2011-04-24": "Easter Sunday (móvil)",
    "2011-04-25": "Easter Monday (móvil)",
    "2011-04-29": "Boda real (solo 2011)",
    "2011-05-02": "Early May bank holiday",
    "2011-05-30": "Spring bank holiday",
    "2011-08-29": "Summer bank holiday",
}


def main():
    df = datos_limpios()
    s_todo = serie_diaria(df, incluir_ultimo=True)

    # 2.1
    n_cal = len(s_todo)
    n_trans = int(s_todo.notna().sum())
    print(f"días calendario {n_cal}, con transacciones {n_trans}")

    # 2.2 días sin transacciones
    faltan = s_todo[s_todo.isna()].index
    tabla = pd.DataFrame({"fecha": faltan.strftime("%Y-%m-%d"),
                          "día": faltan.day_name(locale=None)})
    tabla["clase"] = ["sábado" if d.day_of_week == 5 else
                      FERIADOS.get(d.strftime("%Y-%m-%d"), "cierre de fin de año")
                      for d in faltan]
    no_sabado = tabla[tabla["clase"] != "sábado"]
    guardar_tabla(no_sabado, "p2_dias_faltantes", index=False)
    resumen = tabla["clase"].where(tabla["clase"] == "sábado", "otro").value_counts()
    print(resumen.to_string())
    print(no_sabado.to_string())
    abiertos_dow = s_todo.dropna().index.day_of_week.value_counts().sort_index()
    print("días abiertos por día de la semana (0=lunes):", abiertos_dow.to_dict())

    # 2.4 el último día
    ult = df[df["InvoiceDate"].dt.normalize() == s_todo.index[-1]]
    horas = df.groupby(df["InvoiceDate"].dt.normalize())["InvoiceDate"].max().dt.hour
    print("último día:", s_todo.index[-1].date(), "última hora", ult["InvoiceDate"].max(),
          "venta", round(s_todo.iloc[-1], 2), "| hora de cierre mediana", horas.median(),
          "| venta mediana diaria", round(s_todo.median(), 2))

    resumen21 = pd.DataFrame({"valor": [s_todo.index[0].strftime("%Y-%m-%d"), s_todo.index[-1].strftime("%Y-%m-%d"),
                                        n_cal, n_trans, len(faltan), int((tabla["clase"] == "sábado").sum()),
                                        len(no_sabado)]},
                             index=["primer día", "último día", "días calendario", "días con transacciones",
                                    "días sin transacciones", "  sábados", "  otros"])
    guardar_tabla(resumen21, "p2_resumen", index=True)

    # serie final: sin el último día, días cerrados como NaN
    s = serie_diaria(df)
    train = s[:FIN_TRAIN]
    test = s[INICIO_TEST:FIN_TEST].dropna()
    print(f"entrenamiento: {train.notna().sum()} días abiertos; prueba: {len(test)} días abiertos")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, s.values / 1000, lw=0.8, marker=".", ms=3)
    for d in faltan:
        if d.day_of_week != 5:
            ax.axvline(d, color="red", alpha=0.25, lw=1)
    ax.axvspan(INICIO_TEST, FIN_TEST, color="orange", alpha=0.12, label="prueba")
    ax.set_ylabel("venta neta diaria (miles de £)")
    ax.set_title("Venta neta diaria (líneas rojas: días cerrados que no son sábado)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "p2_serie_diaria.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
