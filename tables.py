from __future__ import annotations
import pandas as pd
import numpy as np

SPANISH_DOW = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# -------------------------------
# 1) Agregado base para tablas
# -------------------------------
def aggregate_for_tables(df_view: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un DataFrame agregado por día/sede/item con UR y UB.
    Columnas:
      - fecha_dt, dia_mes, dow_idx
      - sede_key, id_item
      - UR, UB
    """
    if df_view is None or df_view.empty:
        return pd.DataFrame(columns=["fecha_dt","dia_mes","dow_idx","sede_key","id_item","UR","UB"])

    df = df_view.copy()
    if "fecha_dt" not in df.columns:
        return pd.DataFrame(columns=["fecha_dt","dia_mes","dow_idx","sede_key","id_item","UR","UB"])

    # Normalizaciones mínimas
    df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    df = df[df["fecha_dt"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["fecha_dt","dia_mes","dow_idx","sede_key","id_item","UR","UB"])

    df["dia_mes"] = df["fecha_dt"].dt.day
    df["dow_idx"] = df["fecha_dt"].dt.weekday

    # Columnas clave
    for col, fill in [("sede_key", "S/A"), ("id_item", "S/A")]:
        if col not in df.columns:
            df[col] = fill
    df["id_item"] = df["id_item"].astype(str)
    df["sede_key"] = df["sede_key"].astype(str)

    # Valores
    ur = pd.to_numeric(df.get("und_dia", 0), errors="coerce").fillna(0)
    ub = pd.to_numeric(df.get("ub_unidades", 0), errors="coerce").fillna(0)
    df["und_dia"] = ur
    df["ub_unidades"] = ub

    grp = (df.groupby(["fecha_dt","dia_mes","dow_idx","sede_key","id_item"], as_index=False)
             .agg(UR=("und_dia","sum"), UB=("ub_unidades","sum")))

    grp["UR"] = grp["UR"].astype(int)
    grp["UB"] = grp["UB"].astype(int)
    return grp


# -------------------------------
# 2) Build tabla diaria (pivot)
# -------------------------------
def build_table_from_agg(agg: pd.DataFrame, id_items_sel: list[str], metric: str) -> pd.DataFrame:
    """
    metric: "UR" o "UB".
    Devuelve DataFrame con columnas: Fecha, <sedes...>, T. Día, y una fila final 'Acum. Mes:'.
    """
    if agg is None or agg.empty:
        return pd.DataFrame()

    metric = "UB" if str(metric).upper() == "UB" else "UR"
    val_col = metric

    data = agg[agg["id_item"].isin(id_items_sel)].copy()
    if data.empty:
        return pd.DataFrame()

    # Etiqueta de Fecha: "d/Dow"
    data["Fecha"] = (
        data["dia_mes"].astype(int).astype(str) + "/" +
        data["dow_idx"].astype(int).map(lambda i: SPANISH_DOW[i] if 0 <= i <= 6 else "")
    )

    # Orden de sedes según aparición en los datos
    sede_order = list(pd.Index(data["sede_key"]).drop_duplicates())

    # Pivot
    piv = (data.pivot_table(index="Fecha", columns="sede_key", values=val_col, aggfunc="sum", fill_value=0)
                .reindex(columns=sede_order, fill_value=0))

    # Ordenar filas por día
    idx_as_series = pd.Series(piv.index)
    dia_nums = idx_as_series.str.split("/", n=1, expand=True)[0].astype(int)
    piv = piv.iloc[np.argsort(dia_nums.to_numpy()), :]

    # Total del día
    piv["T. Día"] = piv.sum(axis=1)

    # Reset index
    piv = piv.reset_index()

    # Fila de acumulado del rango
    acum = pd.DataFrame([["Acum. Mes:"] + [int(piv[c].sum()) for c in piv.columns[1:]]], columns=piv.columns)
    out = pd.concat([piv, acum], ignore_index=True)

    # Tipos numéricos
    for c in out.columns:
        if c != "Fecha":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    return out


# -------------------------------
# 3) Style para UI (Styler)
# -------------------------------
def style_table(df_in: pd.DataFrame | pd.io.formats.style.Styler):
    """
    - Resalta 'Acum. Mes:' en negrita.
    - Domingos (Fecha termina en '/Dom') en rojo en toda la fila.
    - Formato entero #,##0 para números.
    Devuelve un pandas Styler.
    """
    if isinstance(df_in, pd.io.formats.style.Styler):
        sty = df_in
        df = df_in.data
    else:
        df = df_in.copy()
        sty = df.style

    if df.empty:
        return sty

    num_cols = [c for c in df.columns if c != "Fecha"]

    def fmt_int(v):
        try:
            return f"{int(v):,}".replace(",", ".")
        except Exception:
            return v

    sty = sty.format({c: fmt_int for c in num_cols})

    def _bold_acum(row):
        return ["font-weight: bold" if str(row.iloc[0]) == "Acum. Mes:" else "" for _ in row]

    def _red_sundays(row):
        fecha = str(row.iloc[0])
        is_dom = fecha.endswith("/Dom")
        return ["color: red" if is_dom else "" for _ in row]

    sty = sty.apply(_bold_acum, axis=1)
    sty = sty.apply(_red_sundays, axis=1)
    sty = sty.set_properties(subset=["Fecha"], **{"width": "90px"})
    return sty
