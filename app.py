# app.py (lite/memory-optimized)
import io
from datetime import timedelta
import gc

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from preprocess import preprocess_cached
from tables import aggregate_for_tables, build_table_from_agg  # (usaremos sin Styler)
from titles import build_title_resumido
from utils import format_df_fast
from ui import inject_css, topbar, kpi_row, chips_row, table_card, chart_card, footer_note

# ===== Config general =====
st.set_page_config(page_title="Ventas x Item – UR / UB (multi-item)", layout="wide")
inject_css()
topbar("Dashboard de UR / UB por sede", "Análisis diario con acumulados")

# Altair: límite de filas para no explotar memoria al serializar a JSON
alt.data_transformers.enable(max_rows=5000)

# ===== Sidebar =====
with st.sidebar:
    st.header("Opciones")
    uploaded = st.file_uploader("Archivo (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    lite_mode = st.toggle("🪶 Modo liviano (recomendado en Cloud)", value=True,
                          help="Desactiva estilos pesados y calcula solo la vista seleccionada.")
    show_dash = st.checkbox("Mostrar '-' cuando sea 0 (solo visual)", value=True)
    debug = st.checkbox("Mostrar diagnóstico", value=False)

if not uploaded:
    st.info("⬅️ Sube un archivo con columnas: empresa, fecha_dcto, id_co, id_item, und_dia (opcional: descripcion, ub_factor, ub_unit)")
    st.stop()

# ===== Lectura + preproceso (cacheado con límites) =====
file_bytes = uploaded.getvalue()

@st.cache_data(show_spinner=True, max_entries=2, ttl=1800)
def _load(bytes_blob, name):
    return preprocess_cached(bytes_blob, name)

try:
    df = _load(file_bytes, uploaded.name)
except Exception as e:
    st.error(str(e))
    st.stop()

# ===== Presets de fecha (anclados a dtmax) =====
def _set_range(days=None, this_month=False, prev_month=False):
    if "fecha_dt" in df and df["fecha_dt"].notna().any():
        dtmax = df["fecha_dt"].max().date()  # último día con datos
        if days is not None:
            st.session_state["_date_range"] = (dtmax - timedelta(days=days-1), dtmax)
        elif this_month:
            start = dtmax.replace(day=1)
            st.session_state["_date_range"] = (start, dtmax)
        elif prev_month:
            first_this = dtmax.replace(day=1)
            last_prev  = first_this - timedelta(days=1)
            start_prev = last_prev.replace(day=1)
            st.session_state["_date_range"] = (start_prev, last_prev)

colp1, colp2, colp3, colp4 = st.columns(4)
with colp1: st.button("Ayer", use_container_width=True, on_click=_set_range, kwargs={"days":1})
with colp2: st.button("7 días", use_container_width=True, on_click=_set_range, kwargs={"days":7})
with colp3: st.button("Este mes", use_container_width=True, on_click=_set_range, kwargs={"this_month":True})
with colp4: st.button("Mes anterior", use_container_width=True, on_click=_set_range, kwargs={"prev_month":True})

# ===== Filtro de rango de fechas =====
if df["fecha_dt"].notna().any():
    min_date = df["fecha_dt"].min().date()
    max_date = df["fecha_dt"].max().date()
    value_range = st.session_state.get("_date_range", (min_date, max_date))
    vr0 = (max(min_date, value_range[0]), min(max_date, value_range[1]))
    rango = st.date_input("Rango de fechas", value=vr0, min_value=min_date, max_value=max_date)
    if isinstance(rango, tuple) and len(rango) == 2:
        d1, d2 = rango
        st.session_state["_date_range"] = (d1, d2)
        # NOTA: no hacemos df.copy() para evitar duplicar memoria si no es necesario
        mask = (df["fecha_dt"].dt.date >= d1) & (df["fecha_dt"].dt.date <= d2)
        df_view = df.loc[mask]
    else:
        df_view = df
else:
    st.info("📅 El archivo no trae fecha completa; el filtro de fechas no está disponible.")
    df_view = df

if df_view.shape[0] == 0:
    st.error("No hay datos en el rango de fechas seleccionado.")
    st.stop()

# ===== Selector de ítems (persistente) =====
if "item_selector" not in st.session_state:
    st.session_state["item_selector"] = []

if "descripcion" in df.columns:
    sum_por_item_full = (df.groupby(["id_item","descripcion"], as_index=True)["und_dia"]
                           .sum().sort_values(ascending=False))
    all_opts = [(i, d) for (i, d) in sum_por_item_full.index]
else:
    sum_por_item_full = df.groupby("id_item", as_index=True)["und_dia"].sum().sort_values(ascending=False)
    all_opts = list(sum_por_item_full.index)

ur_rango = df_view.groupby("id_item")["und_dia"].sum() if df_view.shape[0] else pd.Series(dtype="int64")
solo_con_datos = st.checkbox("Mostrar solo ítems con datos en el rango seleccionado", value=False)

def _tiene_datos_en_rango(item_id: str) -> bool:
    try:
        return ur_rango.get(str(item_id), 0) > 0
    except Exception:
        return False

visible_opts = (
    [(i, d) for (i, d) in all_opts if _tiene_datos_en_rango(i)]
    if (solo_con_datos and len(all_opts) and isinstance(all_opts[0], tuple))
    else ([i for i in all_opts if _tiene_datos_en_rango(i)] if solo_con_datos else all_opts)
)

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

st.multiselect(
    "Items (selecciona 1 a 10)",
    options=visible_opts,
    max_selections=10,
    format_func=fmt,
    key="item_selector"
)

if len(all_opts) and isinstance(all_opts[0], tuple):
    id_items_sel = [str(i) for (i, _d) in st.session_state["item_selector"]]
else:
    id_items_sel = [str(i) for i in st.session_state["item_selector"]]

if not id_items_sel:
    st.warning("Selecciona al menos un item (máximo 10).")
    st.stop()

# ===== Título y KPIs (sobre df_view + selección) =====
df_sel = df_view[df_view["id_item"].isin(id_items_sel)]
titulo_tabla = build_title_resumido(df_view, id_items_sel, top_groups=2)

ur_total = int(df_sel["und_dia"].sum())
ub_total = int(df_sel["ub_unidades"].sum())

# Agregado solo una vez (lo reutilizaremos) para ahorrar CPU/RAM
@st.cache_data(show_spinner=False, max_entries=2, ttl=900)
def _agg_for(df_slice):
    return aggregate_for_tables(df_slice)

agg = _agg_for(df_view)
agg_sel = agg[agg["id_item"].isin(id_items_sel)]
sedes_activas = int((agg_sel.groupby("sede_key")["UR"].sum() > 0).sum())

# Delta vs periodo anterior
if df_view["fecha_dt"].notna().any():
    d1, d2 = df_view["fecha_dt"].min().date(), df_view["fecha_dt"].max().date()
    days = (d2 - d1).days + 1
    prev_start = d1 - timedelta(days=days)
    prev_end   = d1 - timedelta(days=1)
    df_prev = df[(df["fecha_dt"].dt.date >= prev_start) & (df["fecha_dt"].dt.date <= prev_end) & (df["id_item"].isin(id_items_sel))]
    ur_prev, ub_prev = int(df_prev["und_dia"].sum()), int(df_prev["ub_unidades"].sum())
    def _delta(cur, prev):
        if prev == 0: return "—"
        pct = (cur - prev) / prev * 100
        sign = "▲" if pct >= 0 else "▼"
        return f"{sign} {pct:.1f}%"
    ur_delta, ub_delta = _delta(ur_total, ur_prev), _delta(ub_total, ub_prev)
else:
    ur_delta = ub_delta = "—"

kpi_row(ur_total, ub_total, sedes_activas, ur_delta=ur_delta, ub_delta=ub_delta)

# Texto de rango (chips)
if df_view["fecha_dt"].notna().any():
    r1 = df_view["fecha_dt"].min().strftime("%d/%b/%Y")
    r2 = df_view["fecha_dt"].max().strftime("%d/%b/%Y")
    rango_txt = f"{r1} – {r2}"
else:
    rango_txt = "Sin fecha completa"
chips_row(items_count=len(id_items_sel), rango_texto=rango_txt, solo_con_datos=solo_con_datos)

# ===== Navegación por vistas (bajo demanda) =====
view = st.segmented_control(
    "Vistas",
    options=["UR (tabla)", "UB (tabla)", "Día pico", "Top ítems"],
    selection_mode="single",
    default="UR (tabla)",
)

# === Funciones auxiliares con cache y sin copias innecesarias ===
@st.cache_data(show_spinner=False, max_entries=2, ttl=900)
def _table_for_metric(agg_slice, item_ids, metric):
    return build_table_from_agg(agg_slice, item_ids, metric)

# ====== Render condicional (solo lo que se necesita) ======
if view in ("UR (tabla)", "UB (tabla)"):
    metric = "UR" if view.startswith("UR") else "UB"
    tabla = _table_for_metric(agg, id_items_sel, metric)
    if tabla.empty:
        table_card(pd.DataFrame({"Mensaje":[f"Sin datos {metric} para la selección actual."]}), metric, styled=False)
    else:
        # Visual "liviano": sin Styler para ahorrar memoria (activa con lite_mode)
        if lite_mode:
            df_disp = format_df_fast(tabla, show_dash)
            table_card(df_disp, titulo_tabla, styled=False)
        else:
            # Si quieres mantener estilos y tu RAM lo permite, puedes reactivar:
            from tables import style_table
            df_disp = format_df_fast(tabla, show_dash)
            table_card(style_table(df_disp), titulo_tabla, styled=True)

elif view == "Día pico":
    metric_choice = st.radio("Métrica", ["UR", "UB"], horizontal=True, key="metric_pico")
    val_col = "und_dia" if metric_choice == "UR" else "ub_unidades"
    df_metric = df_sel[["fecha_dt","id_item", val_col]].rename(columns={val_col:"valor"})
    if df_metric.shape[0] == 0:
        table_card(pd.DataFrame({"Mensaje":["Sin datos para graficar."]}), f"Día pico – {metric_choice}", styled=False)
    else:
        df_metric["dia"] = df_metric["fecha_dt"].dt.date
        serie = df_metric.groupby("dia", as_index=False)["valor"].sum()
        base = alt.Chart(serie).mark_bar().encode(
            x=alt.X("dia:T", title="Día"),
            y=alt.Y("valor:Q", title=f"{metric_choice}"),
            tooltip=[alt.Tooltip("dia:T", title="Día"), alt.Tooltip("valor:Q", title=metric_choice, format=",.0f")]
        )
        # pico
        if not serie.empty:
            idx_max = serie["valor"].idxmax()
            dia_max, val_max = serie.loc[idx_max, "dia"], int(serie.loc[idx_max, "valor"])
            rule = alt.Chart(pd.DataFrame({"dia": [dia_max]})).mark_rule(color="red").encode(x="dia:T")
            text = alt.Chart(pd.DataFrame({"dia":[dia_max], "txt":[f"Pico: {val_max:,}"]})).mark_text(dy=-10).encode(x="dia:T", text="txt:N")
            chart = base + rule + text
        else:
            chart = base
        chart_card(f"Día pico – {metric_choice}", chart.properties(height=320),
                   filename_base=f"dia_pico_{metric_choice.lower()}",
                   data_df=serie, height=320)

elif view == "Top ítems":
    metric_choice_top = st.radio("Métrica", ["UR", "UB"], horizontal=True, key="metric_top")
    scope = st.radio("Ámbito", ["Todo el rango (todos los ítems)", "Solo ítems seleccionados"], horizontal=True, key="scope_top")
    top_n = st.slider("Top N", min_value=5, max_value=50, value=10, step=5)

    dft = df_sel if scope == "Solo ítems seleccionados" else df_view
    val_col = "und_dia" if metric_choice_top == "UR" else "ub_unidades"

    cols = ["id_item", val_col]
    if "descripcion" in dft.columns:
        cols.append("descripcion")
    dft_small = dft[cols]

    if "descripcion" in dft_small.columns:
        top_df = (dft_small.groupby(["id_item","descripcion"], as_index=False)["und_dia" if metric_choice_top=="UR" else "ub_unidades"]
                    .sum().rename(columns={("und_dia" if metric_choice_top=="UR" else "ub_unidades"):"valor"})
                    .sort_values("valor", ascending=False).head(top_n))
        top_df["Item"] = top_df["descripcion"].where(top_df["descripcion"].astype(str).str.strip() != "", top_df["id_item"])
    else:
        top_df = (dft_small.groupby(["id_item"], as_index=False)["und_dia" if metric_choice_top=="UR" else "ub_unidades"]
                    .sum().rename(columns={("und_dia" if metric_choice_top=="UR" else "ub_unidades"):"valor"})
                    .sort_values("valor", ascending=False).head(top_n))
        top_df["Item"] = top_df["id_item"]

    if top_df.shape[0] == 0:
        table_card(pd.DataFrame({"Mensaje":["Sin datos para ranking."]}), f"Top ítems – {metric_choice_top}", styled=False)
    else:
        chart = alt.Chart(top_df).mark_bar().encode(
            x=alt.X("valor:Q", title=metric_choice_top),
            y=alt.Y("Item:N", sort="-x", title="Ítem"),
            tooltip=[alt.Tooltip("Item:N"), alt.Tooltip("valor:Q", title=metric_choice_top, format=",.0f")]
        ).properties(height=max(250, 24 * len(top_df)))
        chart_card(f"Top {top_n} ítems – {metric_choice_top}", chart,
                   filename_base=f"top_items_{metric_choice_top.lower()}",
                   data_df=top_df, height=max(250, 24 * len(top_df)))

# ===== Export Excel (con estilos ligeros) =====
@st.cache_data(show_spinner=False, max_entries=2, ttl=900)
def to_excel_bytes_styled(df_out: pd.DataFrame, titulo_hoja: str = "reporte") -> bytes:
    df = df_out.copy()
    num_cols = [c for c in df.columns if c != "Fecha"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=titulo_hoja, index=False, startrow=1)
        wb = writer.book; ws = writer.sheets[titulo_hoja]
        fmt_header = wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
        fmt_int, fmt_bold, fmt_red, fmt_red_b, fmt_text = (
            wb.add_format({"num_format": "#,##0"}),
            wb.add_format({"bold": True, "num_format": "#,##0"}),
            wb.add_format({"font_color": "red", "num_format": "#,##0"}),
            wb.add_format({"font_color": "red", "bold": True, "num_format": "#,##0"}),
            wb.add_format()
        )
        for col_idx, col in enumerate(df.columns): ws.write(0, col_idx, col, fmt_header)
        n_rows, n_cols = df.shape
        for r in range(n_rows):
            fecha_val = df.iloc[r, 0]
            is_acum = str(fecha_val) == "Acum. Mes:"
            is_dom  = isinstance(fecha_val, str) and fecha_val.endswith("/Dom")
            for c_idx in range(n_cols):
                val = df.iloc[r, c_idx]
                if c_idx == 0:
                    ws.write(r+1, c_idx, val, (wb.add_format({"bold": True}) if is_acum else wb.add_format({"font_color": "red"}) if is_dom else fmt_text))
                else:
                    if pd.isna(val): ws.write_blank(r+1, c_idx, None)
                    else:
                        ws.write(r+1, c_idx, float(val),
                                 fmt_red_b if (is_acum and is_dom) else fmt_bold if is_acum else fmt_red if is_dom else fmt_int)
        ws.set_column(0, 0, 12)
        for c_idx in range(1, n_cols): ws.set_column(c_idx, c_idx, 12)
    return buf.getvalue()

choice = st.radio("Descargar:", ["UR", "UB"], horizontal=True)
# Construimos la tabla pedida solo aquí para export (si no existe aún)
if choice == "UR":
    tabla_choice = _table_for_metric(agg, id_items_sel, "UR")
else:
    tabla_choice = _table_for_metric(agg, id_items_sel, "UB")
excel_bytes = to_excel_bytes_styled(tabla_choice, titulo_hoja=f"tabla_{choice}")
st.download_button(
    label=f"⬇️ Descargar Excel ({choice}) con estilos",
    data=excel_bytes,
    file_name=f"tabla_ventas_items_{choice.lower()}_styled.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ===== Limpieza (ayuda al GC del contenedor) =====
del file_bytes; gc.collect()

# ===== Debug =====
if debug:
    with st.expander("🔎 Diagnóstico"):
        st.write("Shape df:", df.shape, " | df_view:", df_view.shape, " | df_sel:", df_sel.shape)
        st.write("Fechas (min/max):", str(df["fecha_dt"].min()), "→", str(df["fecha_dt"].max()))
        st.write("IDs seleccionados:", id_items_sel[:10])
        st.write("Memoria df (aprox):", f"{df.memory_usage(deep=True).sum()/1e6:.1f} MB")

footer_note()
