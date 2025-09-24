# preprocess.py
import io
import numpy as np
import pandas as pd
from utils import (
    standardize_columns, unify_empresa, sede_key_to_name,
    parse_dates_strict, extract_day_if_possible, parse_und_dia_series,
    make_base_name
)

from pandas.api.types import is_string_dtype

from typing import Tuple

from constants import SEDE_NAME_MAP

import re

# Parsers de UB (peso/volumen/conteo) para calcular UB por fila
NUM = r"(\d+(?:[.,]\d+)?)"
WEIGHT_RE = re.compile(rf"{NUM}\s*(kg|kilo|kilogramo|kilogramos|kg\.)|{NUM}\s*(g|gr|gramo|gramos|g\.)", re.IGNORECASE)
VOL_RE    = re.compile(rf"{NUM}\s*(l|lt|litro|litros|l\.)|{NUM}\s*(cl|centilitro|centilitros)|{NUM}\s*(ml|mililitro|mililitros)", re.IGNORECASE)
COUNT_RE  = re.compile(r"(?:x\s*(\d{1,5}))|(?:\b(\d{1,5})\s*(?:u|un|und|uds|unid|unids|unidades|huevos?)\b)", re.IGNORECASE)

def _to_float(x):
    try: return float(str(x).replace(",", "."))
    except: return None

def extract_weight_grams(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = WEIGHT_RE.search(text)
    if not m: return None
    if m.group(1):
        kg = _to_float(m.group(1)); return int(round(kg * 1000)) if kg is not None else None
    if m.group(3):
        g = _to_float(m.group(3));  return int(round(g)) if g is not None else None
    return None

def extract_volume_ml(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = VOL_RE.search(text)
    if not m: return None
    if m.group(1):
        l = _to_float(m.group(1));  return int(round(l * 1000)) if l is not None else None
    if m.group(3):
        cl = _to_float(m.group(3)); return int(round(cl * 10)) if cl is not None else None
    if m.group(5):
        ml = _to_float(m.group(5)); return int(round(ml)) if ml is not None else None
    return None

def extract_count_units(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = COUNT_RE.search(text)
    if not m: return None
    for g in (m.group(1), m.group(2)):
        if g:
            try:
                n = int(g)
                return n if 1 <= n <= 5000 else None
            except:
                continue
    return None

# --------- CACHE principal: lee/normaliza todo y deja df listo ---------
import streamlit as st

@st.cache_data(show_spinner=False)
def preprocess_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    # Leer archivo
    if filename.lower().endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))

    # Normalizar columnas (alias → nombres canónicos)
    df = standardize_columns(df_raw)

    required = {"empresa", "fecha_dcto", "id_co", "id_item", "und_dia"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas (después de normalizar): {missing}")

    # Fechas
    parsed = parse_dates_strict(df["fecha_dcto"])
    if parsed.notna().any():
        df["fecha_dt"] = parsed
        df["dia_mes"]  = parsed.dt.day.astype("Int64")
        df["mes_num"]  = parsed.dt.month.astype("Int64")
        df["anio"]     = parsed.dt.year.astype("Int64")
    else:
        df["fecha_dt"] = pd.NaT
        day = extract_day_if_possible(df["fecha_dcto"])
        if day.isna().all():
            raise ValueError("No pude interpretar 'fecha_dcto'. Usa fecha completa (ej. 24/09/2025) o día 1..31.")
        df["dia_mes"] = day
        df["mes_num"] = pd.Series([pd.NA]*len(df), dtype="Int64")
        df["anio"]    = pd.Series([pd.NA]*len(df), dtype="Int64")

    # Normalizaciones clave
    df["empresa"] = df["empresa"].map(unify_empresa)
    idco_num = pd.to_numeric(df["id_co"], errors="coerce")
    df["id_co"] = (idco_num.round().astype("Int64").astype(str)
                   if idco_num.notna().any() else df["id_co"].astype(str))
    df["sede_key"] = df["empresa"].astype(str).str.lower().str.strip() + "|" + df["id_co"].astype(str).str.strip()
    df["id_item"] = df["id_item"].astype(str).str.strip()

    if "descripcion" in df.columns:
        df["descripcion"] = df["descripcion"].astype(str)
        df["desc_base"]   = df["descripcion"].map(make_base_name)
    else:
        df["desc_base"]   = ""

    # UR
    df["und_dia"] = parse_und_dia_series(df["und_dia"])

    # UB por fila
    if "ub_factor" in df.columns:
        ef = pd.to_numeric(df["ub_factor"], errors="coerce").fillna(0)
        ef = ef.where(ef > 0, 0).astype(float)
    else:
        ef = pd.Series(0.0, index=df.index)

    if "ub_unit" in df.columns:
        eu = df["ub_unit"].astype(str).str.lower().str.strip()
        eu = eu.replace({"g":"g","gr":"g","gramo":"g","gramos":"g","g.":"g",
                         "kg":"kg","kilo":"kg","kilogramo":"kg","kilogramos":"kg","kg.":"kg",
                         "ml":"ml","mililitro":"ml","mililitros":"ml",
                         "l":"l","lt":"l","litro":"l","litros":"l","l.":"l"})
    else:
        eu = pd.Series([""]*len(df), index=df.index)

    grams_desc = df["descripcion"].map(extract_weight_grams) if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)
    ml_desc    = df["descripcion"].map(extract_volume_ml)   if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)
    cnt_desc
