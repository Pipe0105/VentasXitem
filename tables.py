import pandas as pd
from constants import SEDE_NAME_MAP
from utils import _order_sede_columns, _fecha_label_from_group


def aggregate_for_tables(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega UR/UB por (día, mes, año, sede, item) y agrega nombre legible de sede.
    Espera que df_in ya esté filtrado por fechas (si aplica).
    """
    agg = (
        df_in.groupby(
            ["dia_mes", "mes_num", "anio", "sede_key", "id_item"],
            as_index=False
        ).agg(
            UR=("und_dia", "sum"),
            UB=("ub_unidades", "sum")
        )
    )

    # Nombre legible de sede a partir de sede_key = "empresa|id_co"
    def _sede_name(k: str) -> str:
        emp, co = str(k).split("|", 1)
        emp, co = emp.strip().lower(), co.strip()
        return SEDE_NAME_MAP.get(emp, {}).get(co, f"{emp}-{co}")

    agg["sede_name"] = agg["sede_key"].map(_sede_name)
    return agg


def build_table_from_agg(agg: pd.DataFrame, id_items_sel: list[str], metric: str) -> pd.DataFrame:
    """
    Construye la tabla final (por día x sede) para la métrica 'UR' o 'UB'.

    - Asegura que salgan TODAS las sedes del mapa (aunque valgan 0).
    - La columna 'Fecha' se muestra como 'día/DOW' (ej. '5/Jue').
    - Agrega columna 'T. Dia' y una fila final 'Acum. Mes:'.
    """
    if not id_items_sel:
        return pd.DataFrame()

    sids = [str(x).strip() for x in id_items_sel]
    dff = agg[agg["id_item"].astype(str).isin(sids)].copy()
    if dff.empty:
        return pd.DataFrame()

    # Mapa día -> mes (moda). Tolerante a NA.
    m = {}
    if "mes_num" in dff.columns and dff["mes_num"].notna().any():
        m = (
            dff.dropna(subset=["dia_mes", "mes_num"])
               .groupby("dia_mes")["mes_num"]
               .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
               .to_dict()
        )

    # Pivot principal
    if metric not in ("UR", "UB"):
        return pd.DataFrame()
    pv = dff.pivot_table(
        index="dia_mes",
        columns="sede_name",
        values=metric,
        aggfunc="sum",
        fill_value=0
    )

    # Asegurar que estén todos los días presentes
    all_days = sorted(dff["dia_mes"].dropna().unique())
    if len(all_days):
        pv = pv.reindex(all_days, fill_value=0)

    # Columnas esperadas (todas las sedes del mapa) + extras detectadas
    full_cols = _order_sede_columns(list(pv.columns))

    # Inyectar columnas faltantes con 0 y reordenar
    for c in full_cols:
        if c not in pv.columns:
            pv[c] = 0
    pv = pv[full_cols]

    # Fecha visible como día/DOW (tolerante a NA)
    pv = pv.reset_index().rename(columns={"dia_mes": "Fecha"})

    # Detectar año y mes predominantes (sin castear a int si hay NaN)
    anio = None
    if "anio" in dff.columns and dff["anio"].notna().any():
        try:
            anio_mode = dff["anio"].mode()
            anio = int(anio_mode.iloc[0]) if not anio_mode.empty and pd.notna(anio_mode.iloc[0]) else None
        except Exception:
            anio = None

    mes = None
    if "mes_num" in dff.columns and dff["mes_num"].notna().any():
        try:
            mes_mode = dff["mes_num"].mode()
            mes = int(mes_mode.iloc[0]) if not mes_mode.empty and pd.notna(mes_mode.iloc[0]) else None
        except Exception:
            mes = None

    pv["Fecha"] = _fecha_label_from_group(
        pv["Fecha"].astype("Int64"),  # serie de días como Int64 (nullable)
        m,                            # dict día -> mes (puede estar vacío)
        anio=anio,
        mes=mes
    )

    # Asegurar enteros y totales por día
    sede_cols = [c for c in pv.columns if c != "Fecha"]
    for c in sede_cols:
        pv[c] = pd.to_numeric(pv[c], errors="coerce").fillna(0).round().astype("Int64")
    pv["T. Dia"] = pv[sede_cols].sum(axis=1).astype("Int64")

    # Fila de acumulado del mes (tolerante a NA)
    acum_values = [pd.to_numeric(pv[c], errors="coerce").fillna(0).sum() for c in sede_cols]
    acum_values = [pd.Series([v]).astype("Int64").iloc[0] for v in acum_values]
    acum_total = pd.to_numeric(pv["T. Dia"], errors="coerce").fillna(0).sum()
    acum_total = pd.Series([acum_total]).astype("Int64").iloc[0]

    acum_row = pd.DataFrame(
        [["Acum. Mes:"] + acum_values + [acum_total]],
        columns=["Fecha"] + sede_cols + ["T. Dia"]
    )

    final = pd.concat([pv, acum_row], ignore_index=True)

    # 🔒 Garantía: máxima 1 fila "Acum. Mes:"
    dup_mask = final["Fecha"].astype(str).eq("Acum. Mes:")
    if dup_mask.sum() > 1:
        final = pd.concat([final[~dup_mask], final[dup_mask].tail(1)], ignore_index=True)

    return final.reset_index(drop=True)


# =================== Estilos para Streamlit ===================
def style_table(df: pd.DataFrame):
    """
    Aplica estilos:
    - Fila de 'Acum. Mes:' → negrita
    - Filas de domingos (Fecha termina en '/Dom') → texto rojo
    """
    def highlight_rows(row):
        if row["Fecha"] == "Acum. Mes:":
            return ["font-weight: bold"] * len(row)
        if isinstance(row["Fecha"], str) and row["Fecha"].endswith("/Dom"):
            return ["color: red"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight_rows, axis=1)
