from __future__ import annotations
import pandas as pd
import numpy as np

from utils import sede_key_to_name, _order_sede_columns, SPANISH_DOW

# Compat: obtener Styler de forma segura
try:
    from pandas.io.formats.style import Styler as _PandasStyler
except Exception:
    _PandasStyler = None

# -------------------------------
# 1) Agregado base para tablas
# -------------------------------
def aggregate_for_tables(df_view: pd.DataFrame) -> pd.DataFrame:
    if df_view is None or df_view.empty:
        return pd.DataFrame(columns=["fecha_dt","dia_mes","dow_idx","sede_key","id_item","UR","UB"])

    df = df_view.copy()
    if "fecha_dt" not in df.columns:
        return pd.DataFrame(columns=["fecha_dt","dia_mes","dow_idx","sede_key","id_item","UR","UB"])

    df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    df = df[df["fecha_dt"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["fecha_dt","dia_mes","dow_idx","sede_key","id_item","UR","UB"])

    df["dia_mes"] = df["fecha_dt"].dt.day
    df["dow_idx"] = df["fecha_dt"].dt.weekday
    for col, fill in [("sede_key", "na|NA"), ("id_item", "S/A")]:
        if col not in df.columns:
            df[col] = fill
    df["id_item"] = df["id_item"].astype(str)
    df["sede_key"] = df["sede_key"].astype(str)

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
    if agg is None or agg.empty:
        return pd.DataFrame()

    metric = "UB" if str(metric).upper() == "UB" else "UR"
    val_col = metric

    data = agg[agg["id_item"].isin(id_items_sel)].copy()
    if data.empty:
        return pd.DataFrame()

    data["Fecha"] = (
        data["dia_mes"].astype(int).astype(str) + "/" +
        data["dow_idx"].astype(int).map(lambda i: SPANISH_DOW[i] if 0 <= i <= 6 else "")
    )

    data["sede_name"] = data["sede_key"].map(sede_key_to_name)
    sede_order = _order_sede_columns(pd.Index(data["sede_name"]).drop_duplicates())

    piv = (data.pivot_table(index="Fecha", columns="sede_name", values=val_col, aggfunc="sum", fill_value=0)
                .reindex(columns=sede_order, fill_value=0))

    idx_as_series = pd.Series(piv.index)
    dia_nums = idx_as_series.str.split("/", n=1, expand=True)[0].astype(int)
    piv = piv.iloc[np.argsort(dia_nums.to_numpy()), :]

    piv["T. Día"] = piv.sum(axis=1)
    piv = piv.reset_index()

    acum = pd.DataFrame([["Acum. Mes:"] + [int(piv[c].sum()) for c in piv.columns[1:]]], columns=piv.columns)
    out = pd.concat([piv, acum], ignore_index=True)

    for c in out.columns:
        if c != "Fecha":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    return out

# -------------------------------
# 3) Style para UI
# -------------------------------
def style_table(df_in):
    is_styler = _PandasStyler is not None and isinstance(df_in, _PandasStyler)

    if is_styler:
        sty = df_in
        df = getattr(df_in, "data", None)
        if df is None:
            df = pd.DataFrame()
    else:
        df = df_in.copy()
        sty = df.style

    if df is None or df.empty:
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
