import io
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from preprocess import preprocess_cached
from tables import aggregate_for_tables, build_table_from_agg, style_table
from titles import build_title_resumido
from utils import format_df_fast
from ui import inject_css, topbar, kpi_row, chips_row, table_card, chart_card, footer_note

# ================= Helpers seguros =================
def _safe_int(v, default=0):
    try:
        return int(v) if pd.notna(v) else default
    except Exception:
        return default

def _safe_sum_int(s: pd.Series, default=0) -> int:
    v = pd.to_numeric(s, errors="coerce").sum()
    return _safe_int(v, default=default)

# ================= Page & Top UI =================
st.set_page_config(page_title="Ventas x Item – UR / UB (multi-item)", layout="wide")
inject_css()
topbar("Dashboard de UR / UB por sede", "Análisis diario con acumulados")

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

# ========= Presets de fecha (anclados a dtmax) =========
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

# ========= Filtro de rango de fechas =========
if df["fecha_dt"].notna().any():
    min_date = df["fecha_dt"].min().date()
    max_date = df["fecha_dt"].max().date()
    value_range = st.session_state.get("_date_range", (min_date, max_date))
    vr0 = (max(min_date, value_range[0]), min(max_date, value_range[1]))
    rango = st.date_input("Rango de fechas", value=vr0, min_value=min_date, max_value=max_date)
    if isinstance(rango, tuple) and len(rango) == 2:
        d1, d2 = rango
        st.session_state["_date_range"] = (d1, d2)
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

# ========= Selector de ítems (persistente) =========
if "item_selector" not in st.session_state:
    st.session_state["item_selector"] = []

# Opciones SIEMPRE desde df (completo), no df_view
if "descripcion" in df.columns:
    sum_por_item_full = (df.groupby(["id_item","descripcion"], as_index=True)["und_dia"]
                           .sum().sort_values(ascending=False))
    all_opts = [(i, d) for (i, d) in sum_por_item_full.index]
else:
    sum_por_item_full = df.groupby("id_item", as_index=True)["und_dia"].sum().sort_values(ascending=False)
    all_opts = list(sum_por_item_full.index)

# Feedback de datos en el rango actual
ur_rango = df_view.groupby("id_item")["und_dia"].sum() if not df_view.empty else pd.Series(dtype="int64")
solo_con_datos = st.checkbox("Mostrar solo ítems con datos en el rango seleccionado", value=False)

def _tiene_datos_en_rango(item_id: str) -> bool:
    try:
        return _safe_int(ur_rango.get(str(item_id), 0)) > 0
    except Exception:
        return False

# Filtrado opcional de opciones visibles
if solo_con_datos:
    if len(all_opts) and isinstance(all_opts[0], tuple):
        visible_opts = [(i, d) for (i, d) in all_opts if _tiene_datos_en_rango(i)]
    else:
        visible_opts = [i for i in all_opts if _tiene_datos_en_rango(i)]
else:
    visible_opts = all_opts

# Etiquetas del multiselect (blindadas)
if len(all_opts) and isinstance(all_opts[0], tuple):
    def fmt(opt):
        i, desc = opt
        ur_full = _safe_int(sum_por_item_full.loc[(i, desc)]) if (i, desc) in sum_por_item_full.index else 0
        ur_rng  = _safe_int(ur_rango.get(str(i), 0))
        badge = f"UR rango: {ur_rng:,}".replace(",", ".") if ur_rng > 0 else "sin datos en rango"
        return f"{i} - {desc}  (UR total: {ur_full:,})  · {badge}".replace(",", ".")
else:
    def fmt(i):
        ur_full = _safe_int(sum_por_item_full.loc[i]) if i in sum_por_item_full.index else 0
        ur_rng  = _safe_int(ur_rango.get(str(i), 0))
        badge = f"UR rango: {ur_rng:,}".replace(",", ".") if ur_rng > 0 else "sin datos en rango"
        return f"{i}  (UR total: {ur_full:,})  · {badge}".replace(",", ".")

# Multiselect con key fija (persiste)
st.multiselect(
    "Items (selecciona 1 a 10)",
    options=visible_opts,
    max_selections=10,
    format_func=fmt,
    key="item_selector"
)

# IDs seleccionados
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

# ========= Formateo de valores para mostrar =========
df_UR_disp = format_df_fast(tabla_UR, show_dash) if not tabla_UR.empty else pd.DataFrame()
df_UB_disp = format_df_fast(tabla_UB, show_dash) if not tabla_UB.empty else pd.DataFrame()

# ========= Título resumido =========
titulo_tabla = build_title_resumido(df_view, id_items_sel, top_groups=2)

# ========= KPIs y delta vs periodo anterior =========
df_sel = df_view[df_view["id_item"].isin(id_items_sel)]
ur_total = _safe_sum_int(df_sel["und_dia"])
ub_total = _safe_sum_int(df_sel["ub_unidades"])

# Sedes activas (UR>0)
agg_sel = agg[agg["id_item"].isin(id_items_sel)]
sedes_activas = _safe_int((agg_sel.groupby("sede_key")["UR"].sum() > 0).sum())

# Delta respecto al periodo anterior de igual largo
if df_view["fecha_dt"].notna().any():
    d1, d2 = df_view["fecha_dt"].min().date(), df_view["fecha_dt"].max().date()
    days = (d2 - d1).days + 1
    prev_start = d1 - timedelta(days=days)
    prev_end   = d1 - timedelta(days=1)
    df_prev = df[(df["fecha_dt"].dt.date >= prev_start) & (df["fecha_dt"].dt.date <= prev_end) & (df["id_item"].isin(id_items_sel))]
    ur_prev = _safe_sum_int(df_prev["und_dia"])
    ub_prev = _safe_sum_int(df_prev["ub_unidades"])

    def _delta(cur, prev):
        if prev == 0: return "—"
        pct = (cur - prev) / prev * 100
        sign = "▲" if pct >= 0 else "▼"
        return f"{sign} {pct:.1f}%"
    ur_delta, ub_delta = _delta(ur_total, ur_prev), _delta(ub_total, ub_prev)
else:
    ur_delta = ub_delta = "—"

kpi_row(ur_total, ub_total, sedes_activas, ur_delta=ur_delta, ub_delta=ub_delta)

# Texto de rango
if df_view["fecha_dt"].notna().any():
    r1 = df_view["fecha_dt"].min().strftime("%d/%b/%Y")
    r2 = df_view["fecha_dt"].max().strftime("%d/%b/%Y")
    rango_txt = f"{r1} – {r2}"
else:
    rango_txt = "Sin fecha completa"

chips_row(items_count=len(id_items_sel), rango_texto=rango_txt, solo_con_datos=solo_con_datos)

# ========= Render en tabs (tablas + gráficos con descarga) =========
tab1, tab2, tab3, tab4 = st.tabs(["🔹 UR (tabla)", "🔸 UB (tabla)", "📈 Día pico", "🏆 Top ítems"])

# ---------- Tabs de tablas ----------
with tab1:
    if not df_UR_disp.empty:
        table_card(style_table(df_UR_disp), titulo_tabla, styled=True)
    else:
        table_card(pd.DataFrame({"Mensaje":["Sin datos UR para la selección actual."]}), "UR", styled=False)

with tab2:
    if not df_UB_disp.empty:
        table_card(style_table(df_UB_disp), titulo_tabla, styled=True)
    else:
        table_card(pd.DataFrame({"Mensaje":["Sin datos UB para la selección actual."]}), "UB", styled=False)

# ---------- 📈 Día pico ----------
with tab3:
    metric_choice = st.radio("Métrica", ["UR", "UB"], horizontal=True, key="metric_pico")
    modo = st.radio("Ver:", ["Agregado selección", "Por ítem"], horizontal=True, key="modo_pico")
    df_metric = df_sel.copy()
    df_metric["dia"] = df_metric["fecha_dt"].dt.date
    val_col = "und_dia" if metric_choice == "UR" else "ub_unidades"

    if modo == "Agregado selección":
        serie = df_metric.groupby("dia", as_index=False)[val_col].sum().rename(columns={val_col: "valor"})
        base = alt.Chart(serie).mark_bar().encode(
            x=alt.X("dia:T", title="Día"),
            y=alt.Y("valor:Q", title=f"{metric_choice}"),
            tooltip=[alt.Tooltip("dia:T", title="Día"), alt.Tooltip("valor:Q", title=metric_choice, format=",.0f")]
        )
        chart = base
        if not serie.empty:
            idx_max = serie["valor"].idxmax()
            dia_max = serie.loc[idx_max, "dia"]
            val_max = _safe_int(serie.loc[idx_max, "valor"])
            rule = alt.Chart(pd.DataFrame({"dia": [dia_max]})).mark_rule(color="red").encode(x="dia:T")
            text = alt.Chart(pd.DataFrame({"dia":[dia_max], "txt":[f"Pico: {val_max:,}"]})).mark_text(dy=-10).encode(x="dia:T", text="txt:N")
            chart = base + rule + text

        chart_card(
            title=f"Día pico – {metric_choice} (agregado selección)",
            chart=chart.properties(height=320),
            filename_base=f"dia_pico_{metric_choice.lower()}",
            data_df=serie,
            height=320
        )

    else:
        serie = (df_metric.groupby(["dia","id_item"], as_index=False)[val_col].sum()
                          .rename(columns={val_col: "valor"}))
        serie_plot = serie.copy()
        if "descripcion" in df_metric.columns:
            id_to_desc = df_metric.groupby("id_item")["descripcion"].agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else s.iloc[0])
            serie_plot["Item"] = serie_plot["id_item"].map(id_to_desc).fillna(serie_plot["id_item"])
        else:
            serie_plot["Item"] = serie_plot["id_item"]

        chart = alt.Chart(serie_plot).mark_line(point=True).encode(
            x=alt.X("dia:T", title="Día"),
            y=alt.Y("valor:Q", title=f"{metric_choice}"),
            color=alt.Color("Item:N", title="Ítem"),
            tooltip=[alt.Tooltip("Item:N"), alt.Tooltip("dia:T", title="Día"), alt.Tooltip("valor:Q", title=metric_choice, format=",.0f")]
        ).properties(height=350)

        chart_card(
            title=f"Evolución diaria por ítem – {metric_choice}",
            chart=chart,
            filename_base=f"evolucion_diaria_{metric_choice.lower()}",
            data_df=serie_plot,
            height=350
        )

# ---------- 🏆 Top ítems ----------
with tab4:
    metric_choice_top = st.radio("Métrica", ["UR", "UB"], horizontal=True, key="metric_top")
    scope = st.radio("Ámbito", ["Todo el rango (todos los ítems)", "Solo ítems seleccionados"], horizontal=True, key="scope_top")
    top_n = st.slider("Top N", min_value=5, max_value=50, value=10, step=5)

    dft = df_sel.copy() if scope == "Solo ítems seleccionados" else df_view.copy()
    val_col = "und_dia" if metric_choice_top == "UR" else "ub_unidades"

    if "descripcion" in dft.columns:
        top_df = (dft.groupby(["id_item","descripcion"], as_index=False)[val_col]
                    .sum().rename(columns={val_col:"valor"})
                    .sort_values("valor", ascending=False).head(top_n))
        top_df["Item"] = top_df["descripcion"].where(top_df["descripcion"].astype(str).str.strip() != "", top_df["id_item"])
    else:
        top_df = (dft.groupby(["id_item"], as_index=False)[val_col]
                    .sum().rename(columns={val_col:"valor"})
                    .sort_values("valor", ascending=False).head(top_n))
        top_df["Item"] = top_df["id_item"]

    chart = alt.Chart(top_df).mark_bar().encode(
        x=alt.X("valor:Q", title=metric_choice_top),
        y=alt.Y("Item:N", sort="-x", title="Ítem"),
        tooltip=[alt.Tooltip("Item:N"), alt.Tooltip("valor:Q", title=metric_choice_top, format=",.0f")]
    ).properties(height=max(250, 24 * len(top_df)))

    chart_card(
        title=f"Top {top_n} ítems – {metric_choice_top}",
        chart=chart,
        filename_base=f"top_items_{metric_choice_top.lower()}",
        data_df=top_df,
        height=max(250, 24 * len(top_df))
    )

# ========= Descarga a Excel (CON estilos) =========
@st.cache_data(show_spinner=False)
def to_excel_bytes_styled(df_out: pd.DataFrame, titulo_hoja: str = "reporte") -> bytes:
    df = df_out.copy()
    num_cols = [c for c in df.columns if c != "Fecha"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=titulo_hoja, index=False, startrow=1)
        wb = writer.book
        ws = writer.sheets[titulo_hoja]

        fmt_header = wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
        fmt_int    = wb.add_format({"num_format": "#,##0"})
        fmt_bold   = wb.add_format({"bold": True, "num_format": "#,##0"})
        fmt_red    = wb.add_format({"font_color": "red", "num_format": "#,##0"})
        fmt_red_b  = wb.add_format({"font_color": "red", "bold": True, "num_format": "#,##0"})
        fmt_text   = wb.add_format()

        for col_idx, col in enumerate(df.columns):
            ws.write(0, col_idx, col, fmt_header)

        n_rows, n_cols = df.shape
        for r in range(n_rows):
            fecha_val = df.iloc[r, 0]
            is_acum   = str(fecha_val) == "Acum. Mes:"
            is_dom    = isinstance(fecha_val, str) and fecha_val.endswith("/Dom")
            for c_idx, col in enumerate(df.columns):
                val = df.iloc[r, c_idx]
                if c_idx == 0:
                    if is_acum:
                        ws.write(r + 1, c_idx, val, wb.add_format({"bold": True}))
                    elif is_dom:
                        ws.write(r + 1, c_idx, val, wb.add_format({"font_color": "red"}))
                    else:
                        ws.write(r + 1, c_idx, val, fmt_text)
                else:
                    if pd.isna(val):
                        ws.write_blank(r + 1, c_idx, None)
                    else:
                        if is_acum and is_dom:
                            ws.write(r + 1, c_idx, float(val), fmt_red_b)
                        elif is_acum:
                            ws.write(r + 1, c_idx, float(val), fmt_bold)
                        elif is_dom:
                            ws.write(r + 1, c_idx, float(val), fmt_red)
                        else:
                            ws.write(r + 1, c_idx, float(val), fmt_int)

        ws.set_column(0, 0, 12)
        for c_idx in range(1, n_cols):
            col_name = str(df.columns[c_idx])
            sample_len = max(len(col_name), 10)
            ws.set_column(c_idx, c_idx, min(14, sample_len + 2))

    return buf.getvalue()

choice = st.radio("Descargar:", ["UR", "UB"], horizontal=True)
df_to_save = tabla_UR if choice == "UR" else tabla_UB
excel_bytes = to_excel_bytes_styled(df_to_save, titulo_hoja=f"tabla_{choice}")
st.download_button(
    label=f"⬇️ Descargar Excel ({choice}) con estilos",
    data=excel_bytes,
    file_name=f"tabla_ventas_items_{choice.lower()}_styled.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ========= Diagnóstico opcional =========
if debug:
    with st.expander("🔎 Diagnóstico"):
        st.write("Fechas (min/max):", str(df["fecha_dt"].min()), "→", str(df["fecha_dt"].max()))
        st.write("Días únicos en vista:", sorted(df_view["dia_mes"].dropna().unique()))
        st.write("IDs seleccionados:", id_items_sel)
        st.write("Muestra df_view:", df_view.head(10))
        st.write("UR por ítem en rango:", ur_rango.head(15))

# ========= Footer =========
footer_note()
