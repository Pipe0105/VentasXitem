# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import preprocess_cached
from tables import aggregate_for_tables, build_table_from_agg
from titles import build_title_resumido
from utils import format_df_fast
from constants import MONTH_ABBR_ES

st.set_page_config(page_title="Ventas x Item – UR / UB (multi-item)", layout="wide")
st.title("📊 Ventas por día y acumulados por sede (UR / UB)")

# ================= Sidebar =================
with st.sidebar:
    st.header("Opciones")
    uploaded = st.file_uploader("Archivo (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    show_dash = st.checkbox("Mostrar '-' cuando sea 0 (solo visual)", value=True)
    debug = st.checkbox("Mostrar diagnóstico", value=False)

if not uploaded:
    st.info("⬅️ Sube un archivo con columnas: empresa, fecha_dcto, id_co, id_item, und_dia (opcional: descripcion, ub_factor, ub_unit)")
    st.stop()

# ========= Lectura + preproceso (cacheado) =========
file_bytes = uploaded.getvalue()
try:
    df = preprocess_cached(file_bytes, uploaded.name)
except Exception as e:
    st.error(str(e))
    st.stop()

# ========= Filtro de rango de fechas (si hay fecha completa) =========
if df["fecha_dt"].notna().any():
    min_date = df["fecha_dt"].min().date()
    max_date = df["fecha_dt"].max().date()
    rango = st.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(rango, tuple) and len(rango) == 2:
        d1, d2 = rango
        mask = (df["fecha_dt"].dt.date >= d1) & (df["fecha_dt"].dt.date <= d2)
        df_view = df.loc[mask].copy()
    else:
        df_view = df.copy()
else:
    st.info("📅 El archivo no trae fecha completa; el filtro de fechas no está disponible.")
    df_view = df.copy()

if df_view.empty:
    st.error("No hay datos en el rango de fechas seleccionado.")
    st.stop()

# ========= Selector de ítems (1..10) =========
if "descripcion" in df_view.columns:
    sum_por_item = (df_view.groupby(["id_item","descripcion"], as_index=True)["und_dia"]
                      .sum().sort_values(ascending=False))
    opts = [(i, d) for (i, d) in sum_por_item.index]
    def fmt(opt):
        i, desc = opt
        return f"{i} - {desc}  (UR: {int(sum_por_item.loc[(i, desc)]):,})".replace(",", ".")
    selected = st.multiselect("Items (selecciona 1 a 10)", opts, max_selections=10, format_func=fmt)
    id_items_sel = [i for (i, _) in selected]
else:
    sum_por_item = df_view.groupby("id_item", as_index=True)["und_dia"].sum().sort_values(ascending=False)
    opts = list(sum_por_item.index)
    def fmt(i):
        return f"{i}  (UR: {int(sum_por_item.loc[i]):,})".replace(",", ".")
    id_items_sel = st.multiselect("Items (selecciona 1 a 10)", opts, max_selections=10, format_func=fmt)

if not id_items_sel:
    st.warning("Selecciona al menos un item (máximo 10).")
    st.stop()

# ========= Agregado para el rango filtrado =========
agg = aggregate_for_tables(df_view)

# ========= Construcción de tablas =========
tabla_UR = build_table_from_agg(agg, id_items_sel, "UR")
tabla_UB = build_table_from_agg(agg, id_items_sel, "UB")

if tabla_UR.empty and tabla_UB.empty:
    st.error("No hay datos para los ítems seleccionados en el rango.")
    st.stop()

# ========= Formateo visual rápido =========
df_UR_disp = format_df_fast(tabla_UR, show_dash) if not tabla_UR.empty else pd.DataFrame()
df_UB_disp = format_df_fast(tabla_UB, show_dash) if not tabla_UB.empty else pd.DataFrame()

# ========= Título resumido (inteligente) =========
titulo_tabla = build_title_resumido(df_view, id_items_sel, top_groups=2)

# ========= Render en tabs =========
tab1, tab2 = st.tabs(["🔹 UR", "🔸 UB"])

with tab1:
    if not df_UR_disp.empty:
        st.subheader(titulo_tabla)
        st.dataframe(df_UR_disp, width="stretch")
    else:
        st.info("Sin datos UR para la selección actual.")

with tab2:
    if not df_UB_disp.empty:
        st.subheader(titulo_tabla)
        st.dataframe(df_UB_disp, width="stretch")
    else:
        st.info("Sin datos UB para la selección actual.")

# ========= Descarga a Excel =========
@st.cache_data(show_spinner=False)
def to_excel_bytes(df_out: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, sheet_name="reporte", index=False)
        wb = writer.book
        ws = writer.sheets["reporte"]
        fmt_int = wb.add_format({"num_format": "#,##0"})
        for col_idx, col in enumerate(df_out.columns):
            ws.set_column(col_idx, col_idx, 12 if col != "Fecha" else 10, None if col=="Fecha" else fmt_int)
    return buf.getvalue()

choice = st.radio("Descargar:", ["UR","UB"], horizontal=True)
df_to_save = tabla_UR if choice == "UR" else tabla_UB
excel_bytes = to_excel_bytes(df_to_save)
st.download_button(
    label=f"⬇️ Descargar Excel ({choice}).xlsx",
    data=excel_bytes,
    file_name=f"tabla_ventas_items_{choice.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ========= Diagnóstico opcional =========
if debug:
    with st.expander("🔎 Diagnóstico"):
        st.write("Fechas (min/max):", str(df["fecha_dt"].min()), "→", str(df["fecha_dt"].max()))
        st.write("Agregado (primeras 20 filas):")
        st.dataframe(agg.head(20), width="stretch")
        st.write("Días únicos en vista:", sorted(df_view["dia_mes"].dropna().unique()))
        st.write("Ejemplo de filas:", df_view.head(10))
