# preprocess.py
from __future__ import annotations
import io
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np


# =========================
# Lectura de archivo
# =========================
def _read_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Lee CSV o Excel desde bytes según la extensión.
    """
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        # encoding='utf-8' por defecto; si el archivo trae latin-1, pandas suele tolerar.
        # Puedes agregar encoding='latin-1' si en tus datos es común.
        df = pd.read_csv(bio)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(bio, engine="openpyxl")
    else:
        # Intento 1 CSV, si falla, Excel
        try:
            bio.seek(0)
            df = pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio, engine="openpyxl")
    return df


# =========================
# Normalización de columnas
# =========================
_COL_CANDIDATES: Dict[str, List[str]] = {
    "fecha_dcto": ["fecha_dcto", "fecha", "fch", "fecha_doc", "fecha_documento", "date"],
    "empresa":    ["empresa", "cia", "compania", "company", "empr"],
    "id_co":      ["id_co", "sede", "co", "centro", "tienda", "almacen"],
    "id_item":    ["id_item", "item", "codigo", "sku", "producto", "id_producto", "codigo_item"],
    "descripcion":["descripcion", "desc", "detalle", "nombre", "producto_desc"],
    "und_dia":    ["und_dia", "unidades", "cantidad", "cant", "q", "qty", "venta_und", "und"],
    "ub_factor":  ["ub_factor", "factor_ub", "factor", "conv_ub", "factor_conversion"],
    "ub_unidades":["ub_unidades", "ub", "unid_base", "unidad_base", "unidades_base"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baja a minúsculas, quita espacios, y mapea a nombres canónicos si hay variantes.
    """
    original_cols = list(df.columns)
    norm_map = {c: str(c).strip().lower().replace(" ", "_") for c in original_cols}
    df = df.rename(columns=norm_map)

    # Mapeo a nombres canónicos
    inv_index: Dict[str, str] = {}
    for canon, cands in _COL_CANDIDATES.items():
        for c in cands:
            if c in df.columns:
                inv_index[canon] = c
                break

    for canon, used in inv_index.items():
        if canon != used:
            df = df.rename(columns={used: canon})

    return df


# =========================
# Fechas: parse robusto
# =========================
# formatos comunes que hemos visto en fuentes reales
_DATE_FORMATS = [
    "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y", "%d-%b-%Y",
    "%m/%d/%Y", "%m-%d-%Y"  # por si llega estilo US
]

def _parse_fecha_series(s: pd.Series) -> pd.Series:
    """
    Parsea fechas de forma robusta sin soltar warnings de 'Could not infer format'.
    Primero intenta formatos conocidos; si no, cae a dateutil (infer).
    """
    s_str = s.astype(str).str.strip().replace({"": np.nan, "NaT": np.nan, "None": np.nan})
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    mask_left = s_str.notna()
    if not mask_left.any():
        return out

    # Intentos con formatos explícitos (rápidos y consistentes)
    remaining = s_str[mask_left]
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(remaining, format=fmt, errors="coerce")
            ok = parsed.notna()
            out.loc[ok.index] = parsed.combine_first(out.loc[ok.index])
            # filtra los que ya parsearon
            remaining = remaining[~ok]
            if remaining.empty:
                break
        except Exception:
            # sigue al siguiente formato
            pass

    # Fallback: parseo flexible (dateutil) con dayfirst=True
    if not remaining.empty:
        parsed_flex = pd.to_datetime(remaining, dayfirst=True, errors="coerce")
        out.loc[remaining.index] = out.loc[remaining.index].combine_first(parsed_flex)

    return out


# =========================
# Cálculo UB y columnas derivadas
# =========================
def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que existan und_dia (enteros) y ub_unidades.
    Si ub_unidades no viene, lo calcula como und_dia * ub_factor (si existe).
    """
    # und_dia
    if "und_dia" not in df.columns:
        raise ValueError("No se encontró la columna de unidades diarias (und_dia). Revisa el archivo de entrada.")

    df["und_dia"] = pd.to_numeric(df["und_dia"], errors="coerce")

    # ub_unidades
    if "ub_unidades" in df.columns:
        df["ub_unidades"] = pd.to_numeric(df["ub_unidades"], errors="coerce")
    else:
        if "ub_factor" in df.columns:
            df["ub_factor"] = pd.to_numeric(df["ub_factor"], errors="coerce").fillna(1.0)
            df["ub_unidades"] = df["und_dia"] * df["ub_factor"]
        else:
            # Si no hay factor, igual creamos la columna (igual a und_dia)
            df["ub_unidades"] = df["und_dia"]

    return df


def _add_date_parts(df: pd.DataFrame, fecha_col: str = "fecha_dcto") -> pd.DataFrame:
    """
    Agrega partes de fecha: fecha_dt, dia_mes, mes_num, anio.
    """
    if fecha_col not in df.columns:
        raise ValueError("No se encontró la columna de fecha (fecha_dcto).")

    fecha_parsed = _parse_fecha_series(df[fecha_col])
    df["fecha_dt"] = fecha_parsed

    # Filtra filas sin fecha válida
    df = df[df["fecha_dt"].notna()].copy()

    df["dia_mes"] = df["fecha_dt"].dt.day.astype("Int64")
    df["mes_num"] = df["fecha_dt"].dt.month.astype("Int64")
    df["anio"]    = df["fecha_dt"].dt.year.astype("Int64")
    return df


def _build_sede_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea sede_key = 'empresa|id_co' en minúsculas; valida columnas.
    """
    if "empresa" not in df.columns or "id_co" not in df.columns:
        raise ValueError("Faltan columnas para sede: se requieren 'empresa' y 'id_co'.")

    df["empresa"] = df["empresa"].astype("string").str.strip().str.lower()
    df["id_co"]   = df["id_co"].astype("string").str.strip()

    df["sede_key"] = (df["empresa"] + "|" + df["id_co"]).astype("string")
    return df


# =========================
# Downcast: ahorro de memoria
# =========================
def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce tipos para ahorrar memoria:
      - numéricos -> int/float más chicos
      - strings frecuentes -> category
    """
    # Númericos a enteros/float más pequeños
    num_like = ["und_dia", "ub_unidades", "ub_factor"]
    for c in num_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # si todos son enteros (o NaN), se puede downcastear a entero
            if pd.isna(df[c]).all():
                continue
            # prueba entero
            c_asint = pd.to_numeric(df[c], errors="coerce", downcast="integer")
            # si la diferencia contra float original es nula (o la col era int de entrada), nos quedamos con entero
            if np.allclose(c_asint.fillna(0).astype(float), df[c].fillna(0).astype(float), equal_nan=True):
                df[c] = c_asint
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")

    # IDs y claves a category (gran ahorro)
    cat_like = ["empresa", "id_co", "id_item", "descripcion", "sede_key"]
    for c in cat_like:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("").astype("category")

    # Fechas y derivadas: ya están en tipos concretos
    # dia_mes, mes_num, anio como enteros pequeños si es posible
    for c in ["dia_mes", "mes_num", "anio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer").astype("Int64")

    return df


# =========================
# Validaciones mínimas
# =========================
def _validate_core(df: pd.DataFrame):
    needed = ["fecha_dt", "empresa", "id_co", "id_item", "und_dia", "ub_unidades", "sede_key"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


# =========================
# Función principal cacheable
# =========================
def preprocess_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Pipeline de preproceso:
      1) Leer CSV/XLSX
      2) Normalizar nombres de columnas
      3) Campos mínimos (empresa, id_co, id_item, fecha_dcto, und_dia, [descripcion], [ub_factor]/[ub_unidades])
      4) Parsear fecha a fecha_dt y descomponer
      5) Asegurar métricas UR/UB
      6) Construir sede_key
      7) Downcast dtypes para reducir memoria

    Retorna un DataFrame listo para usar por el app.
    """
    df = _read_any(file_bytes, filename)
    if df is None or df.shape[0] == 0:
        raise ValueError("El archivo está vacío o no se pudo leer.")

    # Normaliza nombres y hace mapeos canónicos
    df = _normalize_columns(df)

    # Validaciones base
    # Si falta id_item, intenta derivar desde alguna otra columna típica
    if "id_item" not in df.columns:
        raise ValueError("No se encontró la columna del ítem (id_item). Revisa las columnas del archivo.")

    # Fecha + partes
    df = _add_date_parts(df, fecha_col="fecha_dcto")

    # Métricas (UR/UB)
    df = _ensure_metrics(df)

    # Sede key
    df = _build_sede_key(df)

    # Ordena algunas columnas útiles al frente (si existen)
    ordered = [c for c in ["empresa", "id_co", "sede_key", "fecha_dcto", "fecha_dt",
                           "anio", "mes_num", "dia_mes",
                           "id_item", "descripcion", "und_dia", "ub_factor", "ub_unidades"]
               if c in df.columns]
    others = [c for c in df.columns if c not in ordered]
    df = df[ordered + others]

    # Downcast final (ahorro memoria)
    df = _downcast_df(df)

    # Validación final
    _validate_core(df)

    return df
