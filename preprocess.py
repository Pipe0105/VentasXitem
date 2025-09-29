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
# Normalización de texto
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

# Peso
GRAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?)\b", re.IGNORECASE)
KG_RE   = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:kg|kilo?s?)\b", re.IGNORECASE)

# Volumen
ML_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:ml|mililitros?)\b", re.IGNORECASE)
L_RE  = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:l|lt|litros?)\b", re.IGNORECASE)

# Conteo con número: "x12" / "*12" / "12 und|unid|uds"
COUNT_RE = re.compile(
    r"(?:\b|^)(?:x|\*)\s*(\d{1,4})\b"
    r"|(?:\b|^)(\d{1,4})\s*(?:u|un|und|uds|unid|unids|unidades)\b",
    re.IGNORECASE,
)

# SIN número explícito → x1: "*und", "x und", "* unid", etc.
STAR_UND_RE = re.compile(
    r"(?:\*|x)\s*(?:u|un|und|uds|unid|unids|unidades)\b",
    re.IGNORECASE,
)

def _to_float(x: str) -> float | None:
    try:
        return float(str(x).replace(".", "").replace(",", "."))
    except Exception:
        return None

def extract_weight_unit(text: str) -> float | None:
    if not isinstance(text, str): return None
    t = _normalize_text(text)
    m = GRAM_RE.search(t)
    if m:
        v = _to_float(m.group(1))
        return v if v and v > 0 else None
    m = KG_RE.search(t)
    if m:
        v = _to_float(m.group(1))
        return v * 1000 if v and v > 0 else None
    return None

def extract_volume_unit(text: str) -> float | None:
    if not isinstance(text, str): return None
    t = _normalize_text(text)
    m = ML_RE.search(t)
    if m:
        v = _to_float(m.group(1))
        return v if v and v > 0 else None
    m = L_RE.search(t)
    if m:
        v = _to_float(m.group(1))
        return v * 1000 if v and v > 0 else None
    return None

def extract_count_units(text: str) -> int | None:
    """
    Multiplicador de unidades:
      - "x12", "*12", "12 und/unid/uds" → 12
      - "*und", "x und", "* unid", etc. → 1
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
# Cálculo de claves/UB
# =========================

def _build_sede_key(row: pd.Series) -> str:
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
    und_dia = float(row.get("und_dia") or 0)

    # 1) Peso (gramos por unidad)
    if row.get("peso_u"):
        g = float(row["peso_u"])
        return int(round(und_dia / g)) if g > 0 and und_dia > 0 else 0

    # 2) Volumen (ml por unidad)
    if row.get("vol_u"):
        ml = float(row["vol_u"])
        return int(round(und_dia / ml)) if ml > 0 and und_dia > 0 else 0

    # 3) Conteo (incluye "*UND"/"x und" → 1)
    if row.get("count_u"):
        return int(row["count_u"])

    # 4) Fallback
    return int(und_dia)

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

    # Señales desde descripción
    desc = df["descripcion"].fillna("").astype(str)
    df["peso_u"]  = desc.apply(extract_weight_unit)   # en gramos
    df["vol_u"]   = desc.apply(extract_volume_unit)   # en ml
    df["count_u"] = desc.apply(extract_count_units)   # xN o *UND → 1

    # UB
    df["ub_unidades"] = df.apply(_compute_ub_row, axis=1).astype(int)

    # ===== Campos que pide tables.py en el groupby =====
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
