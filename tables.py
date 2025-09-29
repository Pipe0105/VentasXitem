# tables.py
from __future__ import annotations
import math
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None  # Permite usar funciones en entornos sin Streamlit (tests, etc.)

# --------- Compat: detectar Styler sin depender de rutas internas de pandas ----------
try:
    # En varias versiones existe esta importación; si falla, usamos fallback robusto
    from pandas.io.formats.style import Styler as _PandasStyler  # type: ignore
except Exception:
    _PandasStyler = type(pd.DataFrame().style)


def _is_styler(obj) -> bool:
    """Detección robusta de Styler sin depender de pd.io.formats.style."""
    if isinstance(obj, _PandasStyler):
        return True
    # Duck-typing: los Styler tienen .data (DataFrame) y .to_html()
    return hasattr(obj, "data") and hasattr(obj, "to_html")


def _format_number(x, thousand_sep=".", decimal_sep=",", zero_as_dash=False):
    """Formatea números para vista (no muta el DataFrame)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if zero_as_dash and (x == 0 or x == 0.0):
        return "-"
    if isinstance(x, (int,)):
        # Separador de miles
        s = f"{x:,}"
        return s.replace(",", "§").replace(".", thousand_sep).replace("§", thousand_sep)
    if isinstance(x, float):
        s = f"{x:,.2f}"
        s = s.replace(",", "§").replace(".", decimal_sep).replace("§", thousand_sep)
        return s
    return x


def style_table(
    df_in,
    *,
    thousand_sep=".",
    decimal_sep=",",
    zero_as_dash=True,
    precision=2,
    hide_index=True,
):
    """
    Acepta un DataFrame o un Styler y devuelve un Styler con formato.
    - Evita usar pd.io.formats.style.* directamente (compatible con pandas 2.2+ / 2.3+).
    - No altera los datos: solo formato para la vista.
    """
    if _is_styler(df_in):
        sty = df_in  # ya viene como Styler
        df = sty.data
    else:
        df = df_in
        sty = df.style

    # Columnas numéricas para aplicar formato consistente
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Formato numérico con separadores y opción de dash para cero
    if num_cols:
        fmt_func = lambda x: _format_number(
            x,
            thousand_sep=thousand_sep,
            decimal_sep=decimal_sep,
            zero_as_dash=zero_as_dash,
        )
        sty = sty.format({c: fmt_func for c in num_cols}, na_rep="")

    # Estilo básico de tabla (puedes ajustar a tu gusto)
    sty = (
        sty.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("font-weight", "600"), ("text-align", "center")],
                },
                {
                    "selector": "thead th",
                    "props": [("position", "sticky"), ("top", "0"), ("background", "#fafafa")],
                },
                {"selector": "td", "props": [("vertical-align", "middle")]},
            ]
        )
        .set_properties(**{"text-align": "right"})  # números a la derecha por defecto
    )

    # Tratar de ocultar índice si se pide
    if hide_index:
        try:
            # pandas >= 1.4
            sty = sty.hide(axis="index")
        except Exception:
            # Fallback visual si hide no existe
            sty = sty.set_table_styles(
                sty.table_styles
                + [
                    {"selector": ".row_heading", "props": [("display", "none")]},
                    {"selector": "th.blank", "props": [("display", "none")]},
                ]
            )

    return sty


def render_table(
    df_or_styler,
    *,
    use_container_width=True,
    height=None,
    key=None,
):
    """
    Renderiza usando Streamlit. Si no está disponible, retorna HTML del Styler.
    - Para DataFrame: st.dataframe (mejor rendimiento).
    - Para Styler: render por HTML para respetar estilos.
    """
    if st is None:
        # ambiente sin streamlit (tests)
        if _is_styler(df_or_styler):
            return df_or_styler.to_html()
        return df_or_styler

    if _is_styler(df_or_styler):
        html = df_or_styler.to_html()
        st.write(html, unsafe_allow_html=True)
    else:
        st.dataframe(
            df_or_styler,
            use_container_width=use_container_width,
            height=height,
            key=key,
        )


def table_card(
    df_or_styler,
    titulo: str = "",
    *,
    styled: bool = True,
    use_container_width: bool = True,
    height=None,
    key=None,
):
    """
    Card sencilla con título + tabla.
    Compat con llamadas existentes: table_card(style_table(df), titulo_tabla, styled=True)
    """
    if st is None:
        # Sin Streamlit, devolver HTML o DataFrame
        if _is_styler(df_or_styler):
            return df_or_styler.to_html()
        return df_or_styler

    if titulo:
        st.markdown(f"### {titulo}")

    if styled:
        # Si nos pasan DataFrame, lo stilyficamos mínimo para ocultar índice
        obj = df_or_styler if _is_styler(df_or_styler) else style_table(df_or_styler)
        render_table(obj, use_container_width=use_container_width, height=height, key=key)
    else:
        # Render directo de DataFrame
        if _is_styler(df_or_styler):
            render_table(df_or_styler, use_container_width=use_container_width, height=height, key=key)
        else:
            render_table(df_or_styler, use_container_width=use_container_width, height=height, key=key)
