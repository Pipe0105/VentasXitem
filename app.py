# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import preprocess_cached
from tables import aggregate_for_tables, build_table_from_agg, style_table
from titles import build_title_resumido
from utils import format_df_fast

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

# ========= Selector de ítems (persistente, no se reinicia con el rango) =========

# 1) Inicializa estado solo una vez
if "item_selector" not in st.session_state:
    st.session_state["item_selector"] = []

# 2) Construye SIEMPRE las opciones desde el DF COMPLETO (df), no df_view
if "descripcion" in df.columns:
    sum_por_item_full = (df.groupby(["id_item","descripcion"], as_index=True)["und_dia"]
                           .sum().sort_values(ascending=False))
    all_opts = [(i, d) for (i, d) in sum_por_item_full.index]
else:
    sum_por_item_full = df.groupby("id_item", as_index=True)["und_dia"].sum().sort_values(ascending=False)
    all_opts = list(sum_por_item_full.index)

# 3) Info del RANGO ACTUAL para feedback (no para construir opciones)
ur_rango = df_view.groupby("id_item")["und_dia"].sum() if not df_view.empty else pd.Series(dtype="int64")

solo_con_datos = st.checkbox("Mostrar solo ítems con datos en el rango seleccionado", value=False)

def _tiene_datos_en_rango(item_id: str) -> bool:
    try:
        return ur_rango.get(str(item_id), 0) > 0
    except Exception:
        return False

# 4) Filtra opciones visibles si se pide (la selección guardada no se borra)
if solo_con_datos:
    if len(all_opts) and isinstance(all_opts[0], tuple):
        visible_opts = [(i, d) for (i, d) in all_opts if _tiene_datos_en_rango(i)]
    else:
        visible_opts = [i for i in all_opts if _tiene_datos_en_rango(i)]
else:
    visible_opts = all_opts

# 5) Formato que indica UR total y si tiene datos en el rango actual
if len(all_opts) and isinstance(all_opts[0], tuple):
    def fmt(opt):
        i, desc = opt
        ur_full = int(sum_por_item_full.loc[(i, desc)])
        ur_rng = int(ur_rango.get(str(i), 0))
        badge = f"UR rango: {ur_rng:,}".replace(",", ".") if ur_rng > 0 else "sin datos en rango"
        return f"{i} - {desc}  (UR total: {ur_full:,})  · {badge}".replace(",", ".")
else:
    def fmt(i):
        ur_full = int(sum_por_item_full.loc[i]) if i in sum_por_item_full.index else 0
        ur_rng = int(ur_rango.get(str(i), 0))
        badge = f"UR rango: {ur_rng:,}".replace(",", ".") if ur_rng > 0 else "sin datos en rango"
        return f"{i}  (UR total: {ur_full:,})  · {badge}".replace(",", ".")

# 6) Multiselect con key fija → mantiene selección aunque cambie el rango
selected = st.multiselect(
    "Items (selecciona 1 a 10)",
    options=visible_opts,
    max_selections=10,
    format_func=fmt,
    key="item_selector"
)

# 7) Traduce la selección (que puede ser tupla o id) a lista de ids
if len(all_opts) and isinstance(all_opts[0], tuple):
    id_items_sel = [str(i) for (i, _d) in st.session_state["item_selector"]]
else:
    id_items_sel = [str(i) for i in st.session_state["item_selector"]]

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

# ========= Formateo visual rápido (valores) =========
df_UR_disp = format_df_fast(tabla_UR, show_dash) if not tabla_UR.empty else pd.DataFrame()
df_UB_disp = format_df_fast(tabla_UB, show_dash) if not tabla_UB.empty else pd.DataFrame()

# ========= Título resumido (inteligente) =========
titulo_tabla = build_title_resumido(df_view, id_items_sel, top_groups=2)

# ========= Render en tabs (con estilos) =========
tab1, tab2 = st.tabs(["🔹 UR", "🔸 UB"])

with tab1:
    if not df_UR_disp.empty:
        st.subheader(titulo_tabla)
        st.dataframe(style_table(df_UR_disp), use_container_width=True)
    else:
        st.info("Sin datos UR para la selección actual.")

with tab2:
    if not df_UB_disp.empty:
        st.subheader(titulo_tabla)
        st.dataframe(style_table(df_UB_disp), use_container_width=True)
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
        st.write("Días únicos en vista:", sorted(df_view["dia_mes"].dropna().unique()))
        st.write("IDs seleccionados:", id_items_sel)
        st.write("Muestra df_view:", df_view.head(10))
