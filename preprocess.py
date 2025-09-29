import io
import re
import pandas as pd
import streamlit as st

from utils import (
    standardize_columns,
    unify_empresa,
    parse_dates_strict,
    parse_und_dia_series,
)

# =========================
# Regex de presentación
# =========================

# Peso
GRAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?)\b", re.IGNORECASE)
KG_RE   = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:kg|kilo?s?)\b", re.IGNORECASE)

# Volumen
ML_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:ml|mililitros?)\b", re.IGNORECASE)
L_RE  = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:l|lt|litros?)\b", re.IGNORECASE)

# Conteo con número: "x12", "×12", "12 und", "12 unid", "12 uds"
COUNT_RE = re.compile(
    r"(?:\b|^)(?:x|×)\s*(\d{1,4})\b|(?:\b|^)(\d{1,4})\s*(?:u|un|und|uds|unid|unids|unidades)\b",
    re.IGNORECASE,
)

# *** NUEVO: "*und" / "x und" / "×und" SIN número explícito → x1
STAR_UND_RE = re.compile(r"(?:\*|x|×)\s*(?:u|un|und|uds|unid|unids|unidades)\b", re.IGNORECASE)


def _to_float(x: str) -> float | None:
    try:
        return float(str(x).replace(".", "").replace(",", "."))
    except Exception:
        return None


def extract_weight_unit(text: str) -> float | None:
    """Retorna gramos por unidad si hay patrón de peso."""
    if not isinstance(text, str): return None
    m = GRAM_RE.search(text)
    if m:
        v = _to_float(m.group(1))
        return v if v and v > 0 else None
    m = KG_RE.search(text)
    if m:
        v = _to_float(m.group(1))
        return v * 1000 if v and v > 0 else None
    return None


def extract_volume_unit(text: str) -> float | None:
    """Retorna mililitros por unidad si hay patrón de volumen."""
    if not isinstance(text, str): return None
    m = ML_RE.search(text)
    if m:
        v = _to_float(m.group(1))
        return v if v and v > 0 else None
    m = L_RE.search(text)
    if m:
        v = _to_float(m.group(1))
        return v * 1000 if v and v > 0 else None
    return None


def extract_count_units(text: str) -> int | None:
    """Multiplicador de unidades: x12 / 12 und → 12 ; '*und' / 'x und' / '×und' → 1."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = COUNT_RE.search(text)
    if m:
        for g in (m.group(1), m.group(2)):
            if g:
                try:
                    n = int(g)
                    return n if 1 <= n <= 5000 else None
                except Exception:
                    pass
    if STAR_UND_RE.search(text):
        return 1
    return None


def _build_sede_key(row: pd.Series) -> str:
    """
    Construye la clave de sede como 'empresa|id_co'.
    - empresa normalizada con unify_empresa()
    - id_co como texto limpio si existe; si no, usa 'sede' o 'sede_cod'
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
    """UB: prioridad peso → volumen → conteo → fallback und_dia."""
    und_dia = float(row.get("und_dia") or 0)
    peso_u = row.get("peso_u")
    vol_u = row.get("vol_u")
    count_u = row.get("count_u")

    if peso_u:
        g = float(peso_u)
        return int(round(und_dia / g)) if g > 0 and und_dia > 0 else 0
    if vol_u:
        ml = float(vol_u)
        return int(round(und_dia / ml)) if ml > 0 and und_dia > 0 else 0
    if count_u:
        return int(count_u)
    return int(und_dia)


def _preprocess_core(df: pd.DataFrame) -> pd.DataFrame:
    # 1) normaliza nombres
    df = standardize_columns(df)

    # 2) campos básicos
    # fecha_dt
    fecha_src = df.get("fecha_dcto", df.get("fecha", df.get("fecha_dt", None)))
    df["fecha_dt"] = parse_dates_strict(fecha_src) if fecha_src is not None else pd.NaT

    # id_item (string)
    if "id_item" not in df.columns:
        # intenta alguna variante
        for c in df.columns:
            if "item" in str(c).lower():
                df["id_item"] = df[c]
                break
        else:
            df["id_item"] = "S/A"
    df["id_item"] = df["id_item"].astype(str)

    # descripcion
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
        # intenta otras columnas parecidas
        candidate = None
        for c in df.columns:
            if any(k in str(c).lower() for k in ("und", "ur", "cantidad", "cant")):
                candidate = c
                break
        df["und_dia"] = parse_und_dia_series(df.get(candidate, pd.Series(dtype="object"))).fillna(0).astype(int)

    # 3) sede_key
    if "empresa" not in df.columns:
        df["empresa"] = ""
    if "id_co" not in df.columns:
        # intenta otras variantes para código de sede
        if "sede" in df.columns:
            df["id_co"] = df["sede"]
        elif "sede_cod" in df.columns:
            df["id_co"] = df["sede_cod"]
        else:
            df["id_co"] = ""
    df["sede_key"] = df.apply(_build_sede_key, axis=1)

    # 4) señales desde descripción → peso/volumen/conteo
    desc = df["descripcion"].fillna("").astype(str)
    df["peso_u"]  = desc.apply(extract_weight_unit)   # en gramos
    df["vol_u"]   = desc.apply(extract_volume_unit)   # en ml
    df["count_u"] = desc.apply(extract_count_units)   # xN / *und

    # 5) UB
    df["ub_unidades"] = df.apply(_compute_ub_row, axis=1).astype(int)

    # 6) extras para tablas/gráficos
    df["dia_mes"] = df["fecha_dt"].dt.day
    df["dow_idx"] = df["fecha_dt"].dt.weekday

    return df


@st.cache_data(show_spinner=False)
def preprocess_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Entrada: bytes del archivo + nombre. Acepta CSV o Excel.
    Salida: DataFrame con columnas:
      fecha_dt (datetime), id_item (str), descripcion (str),
      empresa (str), id_co (str), sede_key (str),
      und_dia (int), ub_unidades (int),
      dia_mes (int), dow_idx (int)
    """
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        df_raw = pd.read_csv(bio, dtype=str)
    else:
        df_raw = pd.read_excel(bio, dtype=str)
    return _preprocess_core(df_raw)
