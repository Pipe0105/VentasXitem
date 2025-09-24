# ui.py
import streamlit as st
import pandas as pd

APP_CSS = """
<style>
.main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: .2px}
.app-topbar {
  position: sticky; top: 0; z-index: 10;
  display:flex; align-items:center; gap:.75rem; padding:.8rem 1.2rem;
  border:1px solid rgba(0,0,0,.08); border-radius:14px;
  background:linear-gradient(180deg,#ffffff .1%, #fdfdfd);
  box-shadow: 0 6px 14px rgba(0,0,0,.05);
}
.app-badge {
  font-weight:600; font-size:.85rem; color:#2563eb;
  background:#eaf1ff; padding:.25rem .6rem; border-radius:999px;
}
.app-title {font-weight:700; font-size:1.1rem; color:#111827;}
.app-subtle {color:#6b7280; font-size:.9rem}
.card {
  border:1px solid rgba(0,0,0,.06); border-radius:16px; background:#fff; padding:1rem;
  box-shadow: 0 6px 16px rgba(0,0,0,.05);
}
.kpi {text-align:left}
.kpi .value {font-weight:700; font-size:1.6rem; line-height:1}
.kpi .delta {color:#6b7280; font-size:.9rem; margin-top:.2rem}
.kpi .label {color:#6b7280; font-size:.85rem}
.chips {display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0}
.chip {
  border:1px solid #e5e7eb; border-radius:999px; padding:.25rem .7rem; font-size:.85rem; cursor:default;
  background:#f9fafb; color:#374151;
}
.chip.on {background:#eaf1ff; border-color:#c7dbff; color:#2563eb;}
.table-card {padding:.6rem .8rem; border:1px solid #eef2f7; border-radius:16px; background:#ffffff}
.chart-card {padding:.6rem .8rem; border:1px solid #eef2f7; border-radius:16px; background:#ffffff}
.footer {text-align:center; color:#9ca3af; font-size:.85rem; margin-top:1.2rem}
</style>
"""

def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)

def topbar(title: str, subtitle: str = "", badge: str = "Ventas x Ítem"):
    st.markdown(
        f"""
        <div class="app-topbar">
          <div class="app-badge">{badge}</div>
          <div>
            <div class="app-title">{title}</div>
            <div class="app-subtle">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def kpi_row(ur_total: int, ub_total: int, sedes_activas: int, ur_delta: str = "—", ub_delta: str = "—"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card kpi"><div class="label">UR total (rango)</div>'
                    f'<div class="value">{ur_total:,}</div>'
                    f'<div class="delta">{ur_delta}</div></div>'.replace(",", "."),
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card kpi"><div class="label">UB total (rango)</div>'
                    f'<div class="value">{ub_total:,}</div>'
                    f'<div class="delta">{ub_delta}</div></div>'.replace(",", "."),
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card kpi"><div class="label">Sedes con movimiento</div>'
                    f'<div class="value">{sedes_activas:,}</div></div>'.replace(",", "."),
                    unsafe_allow_html=True)

def chips_row(items_count: int, rango_texto: str, solo_con_datos: bool):
    on_cls = "chip on" if solo_con_datos else "chip"
    st.markdown(
        f"""
        <div class="chips">
          <div class="chip">Ítems: <b>{items_count}</b></div>
          <div class="chip">Rango: <b>{rango_texto}</b></div>
          <div class="{on_cls}">Solo con datos</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def table_card(df: pd.DataFrame, title: str, styled=False):
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def chart_card(title: str, chart_renderer):
    """Contenedor visual para gráficos; chart_renderer es una función que dibuja el gráfico."""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader(title)
    chart_renderer()
    st.markdown('</div>', unsafe_allow_html=True)

def footer_note():
    st.markdown('<div class="footer">Hecho con ❤️ sobre Streamlit · UI personalizada</div>', unsafe_allow_html=True)
