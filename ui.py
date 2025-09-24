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
.dl-row {display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.4rem}
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

# ===== Descarga de gráficos (PNG/SVG) con vl-convert-python =====
# Requiere: pip install vl-convert-python
@st.cache_data(show_spinner=False)
def _altair_to_image_bytes(chart_dict: dict, fmt: str = "png", scale: int = 2):
    try:
        import vl_convert as vlc
        if fmt == "png":
            return vlc.vegalite_to_png(chart_dict, scale=scale)
        elif fmt == "svg":
            svg_str = vlc.vegalite_to_svg(chart_dict)
            return svg_str.encode("utf-8")
    except Exception:
        return None
    return None

def chart_card(title: str, chart, filename_base: str, data_df: pd.DataFrame | None = None, height: int | None = None):
    """
    Muestra un gráfico Altair y agrega botones para descargar:
      - PNG (renderizado con vl-convert-python si está instalado)
      - CSV de los datos usados en el gráfico (si se pasa data_df)
    """
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader(title)
    if height:
        chart = chart.properties(height=height)
    st.altair_chart(chart, use_container_width=True)

    # Botones de descarga
    col_a, col_b = st.columns([1,1])
    with col_a:
        chart_dict = chart.to_dict()
        img_png = _altair_to_image_bytes(chart_dict, fmt="png", scale=2)
        if img_png is not None:
            st.download_button(
                "⬇️ Descargar PNG",
                data=img_png,
                file_name=f"{filename_base}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.caption("Para exportar PNG instala `vl-convert-python` en requirements.")
    with col_b:
        if data_df is not None and not data_df.empty:
            csv = data_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar CSV (datos del gráfico)",
                data=csv,
                file_name=f"{filename_base}.csv",
                mime="text/csv",
                use_container_width=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

def footer_note():
    st.markdown('<div class="footer">Hecho con ❤️ sobre Streamlit · UI personalizada</div>', unsafe_allow_html=True)
