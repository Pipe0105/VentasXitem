import io
import re
import unicodedata
import pandas as pd
import streamlit as st

from utils import (
    standardize_columns,
    unify_empresa,
    parse_dates_strict,
    parse_und_dia_series,
)

# =========================
# Normalización ligera
# =========================

_MULT_SIGNS = {
    "×": "x", "✕": "x", "✖": "x", "✗": "x",
    "∙": "x", "·": "x", "•": "x", "∗": "*", "⁎": "*", "٭": "*",
}

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    for k, v in _MULT_SIGNS.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# Regex de presentación
# =========================

# Conteo con número: "x12" / "*12" / "12 und|unid|uds"
COUNT_RE = re.compile(
    r"(?:\b|^)(?:x|\*)\s*(\d{1,4})\b"
    r"|(?:\b|^)(\d{1,4})\s*(?:u|un|und|uds|unid|unids|unidades)\b",
    re.IGNORECASE,
)

# SIN número explícito → x1: "*und", "x und", "* unid", etc. (con o sin espacio)
STAR_UND_RE = re.compile(
    r"(?:\*|x)\s*(?:u|un|und|uds|unid|unids|unidades)\b",
    re.IGNORECASE,
)

def extract_count_multiplier(text: str) -> int | None:
    """
    Factor de unidades por presentación:
      - "x12", "*12", "12 und/unid/uds" → 12
      - "*und", "x und", "* unid", etc. → 1
      - si no hay patrón → None (luego asumimos 1)
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = _normalize_text(text)

    m = COUNT_RE.search(t)
    if m:
        for g in (m.group(1), m.group(2)):
            if g:
                try:
                    n = int(g)
                    return n if 1 <= n <= 5000 else None
                except Exception:
                    pass

    if STAR_UND_RE.search(t):
        return 1

    return None

# =========================
# Claves y UB
# =========================

def _build_sede_key(row: pd.Series) -> str:
    """
    'empresa|codigo' usando:
      - empresa normalizada con unify_empresa()
      - id_co si existe; si no, 'sede' o 'sede_cod'
    """
    emp = unify_empresa(row.get("empresa", ""))
    co = str(row.get("id_co", "")).strip()
    if not co:
        co = str(row.get("sede", "") or row.get("sede_cod", "")).strip()
    if not emp:
        emp = "na"
    if not co:
        co = "NA"
    return f"{emp}|{co}"

def _compute_ub_row(row) -> int:
    """
    UB = und_dia * factor_detectado
    - factor_detectado es:
        * extract_count_multiplier(desc) si existe
        * 1 si no se detecta nada
    """
    und = float(row.get("und_dia") or 0)
    mult = row.get("count_mult")
    if mult is None or mult <= 0:
        mult = 1
    return int(round(und * mult))

# =========================
# Preproceso core + caché
# =========================

def _preprocess_core(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    # fecha_dt
    fecha_src = df.get("fecha_dcto", df.get("fecha", df.get("fecha_dt", None)))
    df["fecha_dt"] = parse_dates_strict(fecha_src) if fecha_src is not None else pd.NaT

    # id_item
    if "id_item" not in df.columns:
        for c in df.columns:
            if "item" in str(c).lower():
                df["id_item"] = df[c]
                break
        else:
            df["id_item"] = "S/A"
    df["id_item"] = df["id_item"].astype(str)

    # descripcion (guardamos original; normalizamos solo para parseo)
    if "descripcion" not in df.columns:
        for c in df.columns:
            if "desc" in str(c).lower():
                df["descripcion"] = df[c]
                break
        else:
            df["descripcion"] = ""
    df["descripcion"] = df["descripcion"].fillna("").astype(str)

    # und_dia (UR)
    if "und_dia" in df.columns:
        df["und_dia"] = parse_und_dia_series(df["und_dia"]).fillna(0).astype(int)
    else:
        candidate = None
        for c in df.columns:
            if any(k in str(c).lower() for k in ("und", "ur", "cantidad", "cant")):
                candidate = c
                break
        df["und_dia"] = parse_und_dia_series(df.get(candidate, pd.Series(dtype="object"))).fillna(0).astype(int)

    # empresa / id_co → sede_key
    if "empresa" not in df.columns:
        df["empresa"] = ""
    if "id_co" not in df.columns:
        if "sede" in df.columns:
            df["id_co"] = df["sede"]
        elif "sede_cod" in df.columns:
            df["id_co"] = df["sede_cod"]
        else:
            df["id_co"] = ""
    df["sede_key"] = df.apply(_build_sede_key, axis=1)

    # ===== Factor de presentación para UB =====
    desc = df["descripcion"].fillna("").astype(str)
    df["count_mult"] = desc.apply(extract_count_multiplier)  # xN o *UND→1 (None → 1 en el cómputo)

    # ===== UB =====
    df["ub_unidades"] = df.apply(_compute_ub_row, axis=1).astype(int)

    # ===== Campos de calendario que usa tables.py =====
    df["dia_mes"] = df["fecha_dt"].dt.day
    df["mes_num"] = df["fecha_dt"].dt.month
    df["anio"]    = df["fecha_dt"].dt.year
    df["dow_idx"] = df["fecha_dt"].dt.weekday

    return df


@st.cache_data(show_spinner=False)
def preprocess_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Lee CSV/XLSX, normaliza y devuelve columnas:
      fecha_dt, id_item, descripcion, empresa, id_co, sede_key,
      und_dia, ub_unidades, dia_mes, mes_num, anio, dow_idx
    """
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        df_raw = pd.read_csv(bio, dtype=str)
    else:
        df_raw = pd.read_excel(bio, dtype=str)
    return _preprocess_core(df_raw)
