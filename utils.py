# utils.py
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from constants import ALIASES, SEDE_NAME_MAP, MONTH_ABBR_ES  # se asume existentes


# ---------------------- Utilidades generales ----------------------
def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas 'Unnamed: *' y devuelve copia."""
    cols = [c for c in df.columns if not (isinstance(c, str) and c.lower().startswith("unnamed"))]
    return df[cols].copy()


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas:
    - strip + lower
    - aplica ALIASES si existe
    - devuelve DataFrame con columnas canónicas
    """
    cols = df.columns.to_series().astype(str).str.strip().str.lower()
    mapping = {}
    for raw in cols:
        canon = normalize_text(raw)
        if canon in ALIASES:
            mapping[raw] = ALIASES[canon]
        else:
            mapping[raw] = canon
    out = df.copy()
    out.columns = [mapping[c] for c in df.columns]
    return out


# ---------------------- Fechas robustas ----------------------
_COMMON_DATE_FORMATS = [
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%Y%m%d",
    "%d.%m.%Y",
    "%d %m %Y",
    "%Y/%m/%d",
]

_COMMON_DATETIME_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
]


def parse_dates_strict(
    s: pd.Series,
    *,
    dayfirst: bool = True,
    keep_time: bool = False,
) -> pd.Series:
    """
    Intenta parsear con formatos conocidos; si falla, usa to_datetime con dayfirst.
    Evita warnings de inferencia y resultados inconsistentes.
    """
    s_str = s.astype(str).str.strip()

    parsed = pd.to_datetime(pd.NaT)  # placeholder

    # Intento 1: formatos explícitos (fecha-hora si keep_time, si no fecha)
    results = None
    fmts = _COMMON_DATETIME_FORMATS + _COMMON_DATE_FORMATS if keep_time else _COMMON_DATE_FORMATS
    for fmt in fmts:
        try:
            cand = pd.to_datetime(s_str, format=fmt, errors="coerce", dayfirst=dayfirst)
            results = cand if results is None else results.fillna(cand)
        except Exception:
            pass

    # Intento 2: to_datetime flexible con dayfirst
    flex = pd.to_datetime(s_str, errors="coerce", dayfirst=dayfirst, infer_datetime_format=False)
    results = flex if results is None else results.fillna(flex)

    return results


# ---------------------- Números ----------------------
def coerce_numbers(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Convierte columnas a numéricas (coerce) respetando separadores comunes.
    """
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)
    for c in cols:
        if c not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        out[c] = (
            out[c]
            .astype(str)
            .str.replace(r"[^\d,\.\-]", "", regex=True)
            .str.replace(".", "", regex=False)  # quita miles estilo es-ES
            .str.replace(",", ".", regex=False)  # decimal a punto
        )
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------- Sedes y orden ----------------------
def order_sedes(df: pd.DataFrame, sede_col: str, region: str) -> pd.DataFrame:
    """
    Ordena la columna de sedes según SEDE_NAME_MAP[region] y mapea códigos -> nombres.
    - region ej.: 'mercamio', 'mtodo', 'bogota'
    """
    out = df.copy()
    region_map = SEDE_NAME_MAP.get(region, {})
    # Mapear códigos a nombres
    out[sede_col] = out[sede_col].astype(str).map(region_map).fillna(out[sede_col].astype(str))
    # Orden categórico según el orden declarado en el dict
    ordered = list(region_map.values())
    if ordered:
        out[sede_col] = pd.Categorical(out[sede_col], categories=ordered, ordered=True)
        out = out.sort_values(by=[sede_col], kind="stable")
    return out


# ---------------------- Fechas legibles ----------------------
SPANISH_DOW = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def add_fecha_leyenda(
    df: pd.DataFrame,
    date_col: str,
    out_col: str = "Fecha",
) -> pd.DataFrame:
    """
    Crea una columna legible 'dd/EEE' (p.ej. 07/Mar) o 'dd/Día' según DOW.
    Requiere que date_col sea datetime64[ns].
    """
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = parse_dates_strict(out[date_col], dayfirst=True)

    # día/semana (abreviado)
    dow = out[date_col].dt.weekday.fillna(0).astype(int).clip(0, 6)
    dia = out[date_col].dt.day.fillna(0).astype(int)
    out[out_col] = dia.astype(str).str.zfill(2) + "/" + dow.map(lambda i: SPANISH_DOW[i])
    return out


# ---------------------- Limpieza principal de DataFrame ----------------------
def preprocess_df(
    df: pd.DataFrame,
    *,
    date_col_candidates: List[str] = ("fecha_dcto", "fecha", "fch", "fecha_doc"),
    sede_col: Optional[str] = None,
    region: Optional[str] = None,
    ensure_no_unnamed: bool = True,
    add_legible_fecha: bool = True,
) -> pd.DataFrame:
    """
    Pipeline de limpieza típico del proyecto:
    - Estandariza nombres de columnas
    - Quita 'Unnamed:*'
    - Convierte fecha
    - Mapea/ordena sedes
    - Elimina índice sin título (se maneja a nivel de estilo en tables.py)
    """
    out = standardize_columns(df)
    if ensure_no_unnamed:
        out = drop_unnamed(out)

    # Detectar columna de fecha
    fecha_col = None
    for c in date_col_candidates:
        if c in out.columns:
            fecha_col = c
            break
    if fecha_col is None:
        # Si no está, intenta detectar por heurística
        for c in out.columns:
            if "fecha" in c:
                fecha_col = c
                break

    if fecha_col is not None:
        out[fecha_col] = parse_dates_strict(out[fecha_col], dayfirst=True)
        if add_legible_fecha:
            out = add_fecha_leyenda(out, fecha_col, out_col="Fecha")

    # Ordenar/mapeo de sedes si procede
    if sede_col and region:
        if sede_col in out.columns:
            out = order_sedes(out, sede_col=sede_col, region=region)

    return out
