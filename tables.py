# tables.py
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
    dff = agg[agg["id_item"].isin(sids)]
    if dff.empty:
        return pd.DataFrame()

    # Mapa "día -> mes (moda)" por si lo necesitas para otros labels (se mantiene)
    m = (
        dff.dropna(subset=["mes_num"])
           .groupby("dia_mes")["mes_num"]
           .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )

    # Pivot principal
    pv = dff.pivot_table(
        index="dia_mes",
        columns="sede_name",
        values=metric,
        aggfunc="sum",
        fill_value=0
    )

    # Asegurar que estén todos los días presentes
    all_days = sorted(dff["dia_mes"].dropna().unique())
    pv = pv.reindex(all_days, fill_value=0)

    # Columnas esperadas (todas las sedes del mapa) + extras detectadas
    full_cols = _order_sede_columns(list(pv.columns))

    # Inyectar columnas faltantes con 0 y reordenar
    for c in full_cols:
        if c not in pv.columns:
            pv[c] = 0
    pv = pv[full_cols]

    # Fecha visible como día/DOW
    pv = pv.reset_index().rename(columns={"dia_mes": "Fecha"})

    # Detectar año y mes predominantes para calcular el día de la semana real
    anio = int(dff["anio"].mode().iloc[0]) if "anio" in dff and dff["anio"].notna().any() else None
    mes  = int(dff["mes_num"].mode().iloc[0]) if "mes_num" in dff and dff["mes_num"].notna().any() else None

    pv["Fecha"] = _fecha_label_from_group(pv["Fecha"], m.to_dict(), anio=anio, mes=mes)

    # Asegurar enteros, totales por día
    sede_cols = [c for c in pv.columns if c != "Fecha"]
    for c in sede_cols:
        pv[c] = pd.to_numeric(pv[c], errors="coerce").fillna(0).round().astype("Int64")
    pv["T. Dia"] = pv[sede_cols].sum(axis=1).astype("Int64")

    # Fila de acumulado del mes
    acum_values = [int(pv[c].sum()) for c in sede_cols]
    acum_total  = int(pv["T. Dia"].sum())
    acum_row = pd.DataFrame(
        [["Acum. Mes:"] + acum_values + [acum_total]],
        columns=["Fecha"] + sede_cols + ["T. Dia"]
    )
    for c in sede_cols + ["T. Dia"]:
        acum_row[c] = pd.to_numeric(acum_row[c], errors="coerce").astype("Int64")

    final = pd.concat([pv, acum_row], ignore_index=True)
    return final.reset_index(drop=True)
